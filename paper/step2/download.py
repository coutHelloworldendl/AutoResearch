import argparse
import requests
import json
import os
import time
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference", type=str, required=True, help="Conference name (e.g., ICLR, NIPS)")
    parser.add_argument("--fields", type=str, required=False, help="List of fields to filter papers by")
    args = parser.parse_args()
    conference_list = args.conference.split(',')
    fields_list = args.fields.split(',') if args.fields else []
    
    with open("/home/squirrel/workspace/AutoResearch/fields/filtered_fields.json", "r") as f:
        valid_fields = [field["field"] for field in json.load(f)]
    for field in fields_list:
        assert field in valid_fields, f"Field '{field}' is not in the list of valid fields."

    for conf in conference_list:
        with open(f"/home/squirrel/workspace/AutoResearch/paper/step2/{conf}.json", "r") as file:
            papers = json.load(file)

        if not os.path.exists(f"/home/squirrel/workspace/AutoResearch/paper/pdf/{conf}"):
            os.makedirs(f"/home/squirrel/workspace/AutoResearch/paper/pdf/{conf}")

        for paper in tqdm(papers):

            if set(fields_list).isdisjoint(set(paper.get("fields", []))):
                continue

            pdf_url = paper.get("pdf_url", None)
            path = f"/home/squirrel/workspace/AutoResearch/paper/pdf/{conf}/{paper['paper_id']}.pdf"
            if os.path.exists(path):
                print("PDF already exists for paper: {}".format(paper.get("title", "Unknown Title")))
                continue

            if pdf_url is None:
                print("No PDF URL for paper: {}".format(paper.get("title", "Unknown Title")))
                continue
            try:
                response = requests.get(pdf_url)
                response.raise_for_status()

                with open(path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)

                print("Downloaded PDF for paper: {}".format(paper.get("paper_id", "Unknown Title")))
                time.sleep(1)  # Be polite and avoid overwhelming the server
            except requests.exceptions.RequestException as e:
                print("Failed to download PDF for paper: {}. Error: {}".format(paper.get("title", "Unknown Title"), str(e)))

    