import os
import json
from typing import Optional, List, Dict
from pathlib import Path

from ...api import API


class Summaryer:
    def __init__(self, field_name: str, step2_dir: str = None, output_file: str = None, api: Optional[API] = None):
        self.field_name = field_name
        self.step2_dir = step2_dir or os.path.join(Path(__file__).resolve().parents[1].as_posix(), "step2")
        self.output_file = output_file or os.path.join(self.step2_dir, f"summary_{self._safe_name(field_name)}.json")
        self.api = api

        with open(os.path.join(Path(__file__).resolve().parents[2].as_posix(), "fields", "filtered_fields.json"), "r") as f:
            self.fields_full = json.load(f)

        self.field_entry = None
        for entry in self.fields_full:
            if entry.get("field") == field_name:
                self.field_entry = entry
                break

        if self.field_entry is None:
            raise ValueError(f"Field '{field_name}' not found in filtered_fields.json")

        # use up to first 5 perspectives (user requested Definition1..5)
        self.perspectives: List[str] = self.field_entry.get("perspectives", [])[:5]

    def _safe_name(self, name: str) -> str:
        return name.replace(" ", "_").replace("/", "_")

    def _build_prompt(self, paper: Dict) -> str:
        # user-specified template, replace Definition1.. with actual perspectives
        defs = []
        for i, p in enumerate(self.perspectives, start=1):
            defs.append(f"({i}) {p}")

        defs_str = ", ".join(defs)

        header = (
            f"Can you analyze the paper contents according to the following perspectives: {defs_str}. "
            "After analysis, please identify each of the perspectives in the paper, and return the answer in the following format: "
            "{\"perspective 1\": plain text, \"perspective 2\": plain text, \"perspective 3\": plain text, ...}"
        )

        # include title, authors, abstract, and pdf_url if available
        parts = [header, "\n\nPaper metadata:\n"]
        parts.append(f"Title: {paper.get('title','')}")
        parts.append(f"Authors: {paper.get('authors','')}")
        if paper.get('pdf_url'):
            parts.append(f"PDF: {paper.get('pdf_url')}")
        parts.append("Abstract:\n" + paper.get('abstract',''))

        return "\n".join(parts)

    def summarize(self, max_per_file: Optional[int] = None) -> List[Dict]:
        results = []

        for fname in os.listdir(self.step2_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(self.step2_dir, fname)
            with open(fpath, 'r') as f:
                try:
                    papers = json.load(f)
                except Exception:
                    continue

            count = 0
            for paper in papers:
                if self.field_name not in paper.get('fields', []):
                    continue

                prompt = self._build_prompt(paper)

                summary_output = None
                if self.api is not None:
                    try:
                        summary_text = self.api.forward(prompt)
                        # 尝试解析为 JSON，否则保留原文
                        try:
                            summary_output = json.loads(summary_text)
                        except Exception:
                            summary_output = {"raw": summary_text}
                    except Exception as e:
                        summary_output = {"error": str(e)}
                else:
                    summary_output = {"note": "api not provided; run with api to generate summaries"}

                out = {
                    "paper_id": paper.get('paper_id'),
                    "title": paper.get('title'),
                    "authors": paper.get('authors'),
                    "pdf_url": paper.get('pdf_url'),
                    "abstract": paper.get('abstract'),
                    "fields": paper.get('fields'),
                    "summary": summary_output,
                }
                results.append(out)

                count += 1
                if max_per_file and count >= max_per_file:
                    break

        # write output
        with open(self.output_file, 'w') as outf:
            json.dump(results, outf, indent=2, ensure_ascii=False)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize papers for a given field from paper/step2 files.")
    parser.add_argument('field', help='Field name as in filtered_fields.json')
    parser.add_argument('--use-api', action='store_true', help='If set, try to call API (requires KIMI_API_KEY env)')
    parser.add_argument('--max-per-file', type=int, default=None, help='Max papers per step2 file to summarize')
    args = parser.parse_args()

    api = None
    if args.use_api:
        # try to construct API wrapper using env var
        api_key = os.environ.get('KIMI_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
        if api_key is None:
            print('Environment API key not found; running without API')
        else:
            # default to kimi if KIMI_API_KEY set, otherwise deepseek
            model = 'kimi' if os.environ.get('KIMI_API_KEY') else 'deepseek'
            api = API(model_name=model)

    summarizer = Summaryer(field_name=args.field, api=api)
    results = summarizer.summarize(max_per_file=args.max_per_file)
    print(f"Wrote {len(results)} summaries to {summarizer.output_file}")
