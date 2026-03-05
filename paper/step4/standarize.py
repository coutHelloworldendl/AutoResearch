import argparse
import json
import os

FIELDS_PATH = "fields/filtered_fields.json"
INPUT_DIR = 'paper/step3'
OUTPUT_DIR = 'paper/step4'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True, help='Field to standardize')
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

    exist_papers = []
    
    for paper in papers:

        summary = paper.get('summary', {})

        if 'error' in summary:
            raise ValueError(f"Paper {paper.get('paper_id')} has error in summary: {summary['error']}")
        
        elif 'raw' in summary:
            raw_text = summary['raw']
            
            text = raw_text.replace('\n', ' ').replace('  ', ' ').replace('```json', '').replace('```', '').strip()
            try:
                standardized_summary = json.loads(text)
                concatenated = ' '.join([f"{key}: {value}" for key, value in standardized_summary.items()])
                paper['summary'] = concatenated
            except Exception as e:
                print(f"Failed to parse summary for paper {paper.get('paper_id')}: {e}") 

        else:

            concatenated = ' '.join([f"{key}: {value}" for key, value in summary.items()])
            paper['summary'] = concatenated
        
        exist_papers.append(paper)

    # Verify

    for paper in exist_papers:
        if 'summary' not in paper:
            raise ValueError(f"Paper {paper.get('paper_id')} missing standardized_summary.")
        elif not isinstance(paper['summary'], str):
            raise ValueError(f"Paper {paper.get('paper_id')} has non-string standardized_summary.")
        elif len(paper['summary'].strip()) == 0:
            raise ValueError(f"Paper {paper.get('paper_id')} has empty standardized_summary.")

    with open(out_file_path, 'w') as f:
        json.dump(exist_papers, f, indent=2)

    print("Saving standardized summaries...")
    