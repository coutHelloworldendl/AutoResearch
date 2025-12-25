import json
import argparse
import os
import random
from api import API

KEYWORDS_DIR = '/home/squirrel/workspace/AutoResearch/paper/step7'
ABSTRACTS_DIR = '/home/squirrel/workspace/AutoResearch/paper/raw'
OUTPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/final'
FIELDS_PATH = "/home/squirrel/workspace/AutoResearch/fields/filtered_fields.json"

prompt = \
'''

You are an experienced research analyst in {field}. Based on the following clustering analysis results of research papers, generate a structured domain survey report.

## CLUSTERING RESULTS:
{clusters_info}

## OUTPUT REQUIREMENTS:

### 1. STRUCTURE FORMAT:
- Output: **Markdown table only** (no additional text)
- Table columns: `Category | Sub-category | What is covered | Cluster`
- Structure: **5-8 main categories**, each with **2-4 sub-categories**
- Numbering: Use hierarchical format (1, 1.1, 1.2, 2, 2.1, etc.)

### 2. CONTENT GUIDELINES:

**For each column:**
- **Category**: Broad research themes (e.g., "Perception & Mapping", "Generative Models")
- **Sub-category**: Specific research directions within each theme
- **What is covered**: 1-sentence description of techniques/applications
- **Cluster**: Relevant cluster IDs (comma-separated)

**Organization principles:**
- Group similar clusters logically
- Prioritize prominent/emerging research areas
- Ensure each cluster appears at least once
- Maintain distinct categories with minimal overlap

### 3. EXAMPLE FORMAT:

| Category | Sub-category | What is covered | Cluster |
|----------|--------------|------------------|---------|
| **1. Perception & Mapping** | | | |
| | 1.1 Multimodal sensor fusion | Fusing heterogeneous sensors for richer scene understanding | 0,6,7,8,14 |
| | 1.2 3D reconstruction | Building geometric maps for localization and navigation | 0,8,16 |
| **2. Manipulation & Grasping** | | | |
| | 2.1 Dexterous manipulation | Multi-finger in-hand manipulation with dexterous hands | 11,12 |

### 4. QUALITY REQUIREMENTS:
- Technical accuracy
- Clear, concise descriptions
- Complete cluster coverage
- No introductory/concluding text

Now generate the survey report based on the clustering results.

'''

# def extract_abstract(paper_id):
#     conference, id = paper_id.split('-')
#     id = int(id)
#     abstracts_file = os.path.join(ABSTRACTS_DIR, f"{conference}.json")
#     if not os.path.exists(abstracts_file):
#         raise ValueError(f"Abstracts file for conference '{conference}' not found.")
#     with open(abstracts_file, 'r') as f:
#         papers = json.load(f)

#     paper = next((p for p in papers if p.get('paper_id') == paper_id), None)
#     if not paper:
#         raise ValueError(f"Paper ID '{paper_id}' not found in {abstracts_file}")
    
#     return paper.get('abstract', '')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True, help='Field to summarize')
    args = parser.parse_args()

    with open(FIELDS_PATH, 'r') as f:
        fields = json.load(f)
    entry = next((e for e in fields if e.get('field') == args.field), None)
    if not entry:
        raise ValueError(f"Field '{args.field}' not found in {FIELDS_PATH}")
    abbr = entry.get('abbr')

    clusters_file = os.path.join(KEYWORDS_DIR, f"{abbr}.json")
    output_file = os.path.join(OUTPUT_DIR, f"{abbr}_survey_report.md")

    with open(clusters_file, 'r') as f:
        clusters = json.load(f)

    clusters_info = ""
    for cluster in clusters:
        cluster_id = cluster.get('cluster')
        summary = cluster.get('summary')

        clusters_info += f"### Cluster {cluster_id}\nSummary: {summary}\n\n"
    
    final_prompt = prompt.format(field=args.field, clusters_info=clusters_info)

    api = API()
    response = api.forward(final_prompt)

    with open(output_file, 'w') as f:
        f.write(response)