import os
import json
import argparse
from tqdm import tqdm
from api import API
import sys

FIELDS_PATH = "/home/squirrel/workspace/AutoResearch/fields/filtered_fields.json"
STEP2_DIR = "/home/squirrel/workspace/AutoResearch/paper/step2"
OUT_DIR = "/home/squirrel/workspace/AutoResearch/paper/step3"


def build_prompt(paper, perspectives):
    defs = ", ".join(f"({i}) {p}" for i, p in enumerate(perspectives, start=1))
    header = f"Can you analyze the paper according to the following perspectives: {defs}. Return JSON mapping each perspective."
    parts = [header, f"Paper Title: {paper.get('title','')}"]
    parts.append("Paper Abstract:\n" + paper.get('abstract', ''))
    parts.append("After analysis, please identify each of the perspectives in the paper, and return the answer in the following format: \n ")
    parts.append("{" + ", ".join(f"{p}: plain text" for p in perspectives) + "}")
    return "\n".join(parts)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True, help='Field to summarize')
    parser.add_argument('--api', choices=['kimi', 'deepseek'], default='deepseek', help='API to use')
    args = parser.parse_args()

    with open(FIELDS_PATH, 'r') as f:
        fields = json.load(f)

    entry = next((e for e in fields if e.get('field') == args.field), None)
    if not entry:
        raise ValueError(f"Field '{args.field}' not found in {FIELDS_PATH}")

    perspectives = entry.get('perspectives', [])[:5]
    abbr = entry.get('abbr')
    api = API(model_name=args.api)

    out_file = os.path.join(OUT_DIR, f"{abbr}.json")
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            results = json.load(f)
            exist_ids = set(r['paper_id'] for r in results)
    else:
        results = []
        exist_ids = set()
    
    collect_papers = []
    for fn in os.listdir(STEP2_DIR):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(STEP2_DIR, fn)
        papers = json.load(open(p, 'r'))
        for paper in papers:
            if args.field not in paper.get('fields', []):
                continue
            collect_papers.append(paper)
        
    print(f"Found {len(collect_papers)} papers for field '{args.field}'")

    for paper in tqdm(collect_papers):
        if paper.get('paper_id') in exist_ids:
            continue

        prompt = build_prompt(paper, perspectives)

        try:
            txt = api.forward(prompt)
            print(f"API Response: {txt}")
            sys.stdout.flush()
            try:
                summary = json.loads(txt)
            except Exception:
                summary = {'raw': txt}
        except Exception as e:
            summary = {'error': str(e)}

        results.append({
            'paper_id': paper.get('paper_id'),
            'title': paper.get('title'),
            'authors': paper.get('authors'),
            'fields': paper.get('fields', []),
            'summary': summary,
        })

        exist_ids.add(paper.get('paper_id'))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_file, 'w') as out:
        json.dump(results, out, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} summaries to {out_file}")



