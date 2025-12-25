import argparse
import json
import os

INPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/step6'
OUTPUT_DIR = '/home/squirrel/workspace/AutoResearch/paper/step7'
FIELDS_PATH = "/home/squirrel/workspace/AutoResearch/fields/filtered_fields.json"

import random
import re
from collections import Counter
from api import API
import sys


def extract_keywords(text, topk=3):
	if not text:
		return []
	words = re.findall(r"[A-Za-z]+", text.lower())
	stopwords = {
		'the', 'and', 'of', 'to', 'in', 'a', 'for', 'is', 'on', 'with', 'as', 'by', 'an', 'are', 'that', 'from',
		'we', 'this', 'be', 'using', 'use', 'used', 'will', 'which', 'our', 'can', 'it', 'its', 'at', 'have'
	}
	filtered = [w for w in words if w not in stopwords and len(w) > 1]
	if not filtered:
		return []
	counts = Counter(filtered)
	return [w for w, _ in counts.most_common(topk)]


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

	abbr = entry.get('abbr')
	in_file_path = os.path.join(INPUT_DIR, f"{abbr}.json")
	out_file_path = os.path.join(OUTPUT_DIR, f"{abbr}.json")

	with open(in_file_path, 'r') as f:
		papers = json.load(f)

	clusters = {}
	for paper in papers:
		cid = paper.get('cluster')
		if cid is None:
			continue
		clusters.setdefault(int(cid), []).append(paper)

	api = API(model_name=args.api)

	results = []
	for cid, plist in sorted(clusters.items()):
		sample_n = min(50, len(plist))
		print(f"Processing cluster {cid} with {len(plist)} papers, sampling {sample_n}")
		sampled = random.sample(plist, sample_n) if sample_n > 0 else []
		abstracts = [p.get('summary', '') for p in sampled if p.get('summary')]
		concat = "\n\n".join(abstracts)

		if not concat:
			summary = ""
		else:
			prompt = "Can you summarize the following contents into three distinct keywords:\n\n" + concat + "please return the keywords wrapped in **double asterisks**, example: **keyword1**, **keyword2**, **keyword3**.\n\n"
			try:
				txt = api.forward(prompt)
				print(f"API Response (cluster {cid}): {txt}")
				sys.stdout.flush()
				# 提取所有用 **...** 包裹的加粗片段，作为关键词（最多取前三个）
				txt = txt or ''
				bolds = re.findall(r"\*\*(.*?)\*\*", txt, flags=re.DOTALL)
				bolds = [b.strip() for b in bolds if b.strip()]
				if bolds:
					summary = ','.join(bolds)
				else:
					summary = ''
			except Exception:
				summary = ''

		print(f"Cluster {cid} summary: {summary}")
		sys.stdout.flush()

		results.append({
			'cluster': int(cid),
			'summary': summary,
			'paper_ids': [p.get('paper_id') for p in plist]
		})

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	with open(out_file_path, 'w') as f:
		json.dump(results, f, ensure_ascii=False, indent=2)



