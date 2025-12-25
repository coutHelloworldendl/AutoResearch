import argparse
import json
from sklearn.cluster import KMeans
import os

FIELDS_PATH = "/home/squirrel/workspace/AutoResearch/fields/filtered_fields.json"
INPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/step5'
OUTPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/step6'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True, help='Field to cluster')
    args = parser.parse_args()

    with open(FIELDS_PATH, 'r') as f:
        fields = json.load(f)

    entry = next((e for e in fields if e.get('field') == args.field), None)
    if not entry:
        raise ValueError(f"Field '{args.field}' not found in {FIELDS_PATH}")

    abbr = entry.get('abbr')
    in_file_path = os.path.join(INPUT_DIR, f"{abbr}.json")
    out_file_path = os.path.join(OUTPUT_DIR, f"{abbr}.json")

    with open(in_file_path, 'r') as f:
        papers = json.load(f)

    embeddings = [paper.get('embedding') for paper in papers]

    if not embeddings:
        raise ValueError(f"No embeddings found in {in_file_path}")

    kmeans = KMeans(n_clusters=30, random_state=0)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    for paper, label in zip(papers, labels):
        paper['cluster'] = int(label)
        del paper['embedding']

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_file_path, 'w') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)