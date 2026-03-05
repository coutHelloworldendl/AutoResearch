import argparse
import requests
import json
import os
import time
from tqdm import tqdm

def download_papers(url, path):
    response = requests.get(url)
    response.raise_for_status()

    with open(path, "wb") as pdf_file:
        for chunk in response.iter_content(chunk_size=8192):
            pdf_file.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference", type=str, required=True, help="Conference name (e.g., ICLR, NIPS)")
    parser.add_argument("--fields", type=str, required=False, help="List of fields to filter papers by")
    args = parser.parse_args()
    conference_list = args.conference.split(',')
    fields_list = args.fields.split(',') if args.fields else []
    
    with open("fields/filtered_fields.json", "r") as f:
        valid_fields = [field["field"] for field in json.load(f)]
    for field in fields_list:
        assert field in valid_fields, f"Field '{field}' is not in the list of valid fields."

    download_list = []

    for conf in conference_list:
        with open(f"paper/step2/{conf}.json", "r") as file:
            papers = json.load(file)

        if not os.path.exists(f"paper/pdf/{conf}"):
            os.makedirs(f"paper/pdf/{conf}")

        for paper in tqdm(papers):

            if set(fields_list).isdisjoint(set(paper.get("fields", []))):
                continue

            if 'agents' not in set(paper.get("abstract", []).lower().split()):
                continue

            path = f"paper/pdf/{conf}/{paper['paper_id']}.pdf"
            if os.path.exists(path):
                print("PDF already exists for paper: {}".format(paper.get("title", "Unknown Title")))
                continue

            pdf_url = paper.get("pdf_url", None)
            if pdf_url is None:
                print("No PDF URL for paper: {}".format(paper.get("title", "Unknown Title")))
                continue

            download_list.append((pdf_url, path))

            print("Prepared to download paper: {}".format(paper.get("title", "Unknown Title")))
            
    print(f"Total papers to download: {len(download_list)}")

    

    # for url, path in tqdm(download_list):
    #     try:
    #         download_papers(url, path)
    #         time.sleep(1)  # Be polite to the server
    #     except Exception as e:
    #         print(f"Failed to download {url}. Error: {e}")

    