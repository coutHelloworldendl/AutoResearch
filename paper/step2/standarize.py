import json
import argparse

class Standardizer:
    def __init__(self, conference_name, year):
        self.conference_name = conference_name
        self.year = year

        with open("fields/filtered_fields.json", "r") as file:
            self.fields_list_full = json.load(file)
        self.fields_list = [field["field"] for field in self.fields_list_full]

    def standardize(self):
        with open(f"paper/step1/{self.conference_name}{self.year}.json", "r") as file:
            papers = json.load(file)

        for paper in papers:
            if paper.get("fields", None) is None:
                raise ValueError("Paper missing fields: {}".format(paper))
            if len(paper["fields"]) > 3:
                print("Paper has more than 3 fields: {}".format(paper["paper_id"]))
            
            paper["fields"] = list(set(paper["fields"]))

            for field in paper["fields"]:
                if field in self.fields_list:
                    continue

                std_field = ''
                if "self-supervised learning" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "machine learning" in field.lower() and "auto" not in field.lower():
                    std_field = "Machine Learning (ML)"
                    if std_field != field:
                        paper["fields"].remove(field)
                        paper["fields"].append(std_field)
                    continue

                if "anomaly detection" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "differential privacy" in field.lower():
                    std_field = "AI Safety & Alignment"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "via normalizing flow" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "federated" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "optimization" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "variational inference" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "multi-task learning" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "conformal prediction" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "Topological Deep Learning".lower() in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "dataset distillation" in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "Neurosymbolic".lower() in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "Graph Neural Networks".lower() in field.lower():
                    std_field = "Machine Learning (ML)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                if "Foundation Models".lower() in field.lower():
                    std_field = "Deep Learning (DL)"
                    paper["fields"].remove(field)
                    paper["fields"].append(std_field)
                    continue

                for candidate_field in self.fields_list:
                    if field.lower() in candidate_field.lower():
                        if std_field != '':
                            raise ValueError("Multiple matching fields found for field '{}' in paper {}".format(field, paper["paper_id"]))
                        std_field = candidate_field

                if std_field == '':
                    raise ValueError("No matching field found for field '{}' in paper {}".format(field, paper["paper_id"]))
                
                paper["fields"].remove(field)
                paper["fields"].append(std_field)

        with open(f"paper/step2/{self.conference_name}{self.year}.json", "w") as file:
            json.dump(papers, file, indent=4)
            print("Standardization complete and saved to file.")

class Verifier:
    def __init__(self, conference_name, year, api):
        self.conference_name = conference_name
        self.year = year
        self.api = api

        with open("fields/filtered_fields.json", "r") as file:
            self.fields_list_full = json.load(file)
        self.fields_list = [field["field"] for field in self.fields_list_full]

    def verify(self):
        with open(f"paper/step2/{self.conference_name}{self.year}.json", "r") as file:
            papers = json.load(file)

        for paper in papers:
            if paper.get("fields", None) is None:
                raise ValueError("Paper missing fields: {}".format(paper))
            if len(paper["fields"]) > 3:
                raise ValueError("Paper has more than 3 fields: {}".format(paper["paper_id"]))
            if len(set(paper["fields"])) != len(paper["fields"]):
                raise ValueError("Paper has duplicate fields: {}".format(paper["paper_id"]))
            for field in paper["fields"]:
                if field not in self.fields_list:
                    raise ValueError("Paper has unknown field '{}' : {}".format(field, paper["paper_id"]))
        
        print("All papers verified successfully.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize and verify paper fields")
    parser.add_argument("--conference_name", type=str, required=True, help="Name of the conference (e.g., NeurIPS, ICML)")
    parser.add_argument("--year", type=int, required=True, help="Year of the conference (e.g., 2023)")
    args = parser.parse_args()
    print("Processing conference:{}, {}".format(args.conference_name, args.year))
    standardizer = Standardizer(conference_name=args.conference_name, year=args.year)
    standardizer.standardize()
    verifier = Verifier(conference_name=args.conference_name, year=args.year, api=None)
    verifier.verify()