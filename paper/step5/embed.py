import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from api import Embed
import argparse
import json
import os
import sys
from tqdm import tqdm

FIELDS_PATH = "/home/squirrel/workspace/AutoResearch/fields/filtered_fields.json"
INPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/step4'
OUTPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/step5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True, help='Field to embed')
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

    with open(out_file_path, 'r') as f_out:
        exist_papers = json.load(f_out)
    exist_paper_ids = {paper['paper_id'] for paper in exist_papers}

    embedder = Embed(model_path='/model/squirrel/NV-Embed-v2')
    for paper in tqdm(papers):

        if paper['paper_id'] in exist_paper_ids:
            continue

        summary = paper.get('summary')
        if not summary:
            raise ValueError(f"Paper {paper.get('paper_id')} has empty summary.")
        
        if not isinstance(summary, str):
            print("[WARNING] Summary {} is not string, converting to string.".format(paper.get('paper_id')))
            sys.stdout.flush()
            summary = str(summary)
        
        embedding = embedder.encode([summary])[0]
        paper['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
        
        exist_papers.append(paper)

    with open(out_file_path, 'w') as f:
        json.dump(exist_papers, f, indent=2)