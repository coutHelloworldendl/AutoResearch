import json
from api import API
from tqdm import tqdm
import random

class Field_Selector:
    def __init__(self, conference_name, year):
        self.conference_name = conference_name
        self.year = year
        self.api = API()
        with open("fields/filtered_fields.json", "r") as file:
            self.fields_list_full = json.load(file)
        self.fields_list = [field["field"] for field in self.fields_list_full]

        self.first_prompt = \
        '''
        You are an expert in academic research papers. Here is a list of research fields:{fields_list}. Given the following abstract of a research paper, identify the 3 most precise and relevant research field from the list. Output only the original names of the fields, separated by commas.
        Abstract: {abstract}
        '''

        self.second_prompt = \
        '''
        You are an expert in academic research papers. Please decide whether the following abstract of the paper precisely and accurately fits into the research field of {field}, which is described as {field_explanation}. Please only answer with "Yes" or "No".
        Abstract: {abstract}
        '''

        self.third_prompt = \
        '''
        You are an expert in academic research papers. Here is a list of research fields:{fields_list}. Given the following abstract of a research paper, identify the most precise and relevant research field from the list. Output only the original name of the field.
        Abstract: {abstract}
        '''
        
    def forward(self):
        with open("paper/raw/{conference_name}{year}.json".format(conference_name=self.conference_name, year=self.year), "r") as file:
            papers = json.load(file)

            for paper in tqdm(papers):
                final_fields = []

                abstract = paper["abstract"]
                prompt = self.first_prompt.format(fields_list=", ".join(self.fields_list), abstract=abstract)
                response = self.api.forward(prompt)
                try:
                    selected_fields = [field.strip() for field in response.split(",")]
                except:
                    print("Error parsing fields from response: {}".format(response))
                    paper["fields"] = final_fields
                    continue

                for field_name in selected_fields:
                    field = next((f for f in self.fields_list_full if field_name.lower() in f["field"].lower()), None)
                    if field is None:
                        print("Field {} not found in fields list".format(field_name))
                        continue
                    field_explanation = field["explanation"]
                    prompt = self.second_prompt.format(field=field_name, field_explanation=field_explanation, abstract=abstract)
                    response = self.api.forward(prompt)
                    if "yes" in response.lower():
                        final_fields.append(field["field"])

                if len(final_fields) == 0:
                    prompt = self.third_prompt.format(fields_list=", ".join(self.fields_list), abstract=abstract)
                    response = self.api.forward(prompt)
                    try:
                        field_name = response.strip()
                        final_fields.append(field_name)
                    except:
                        raise ValueError("Error parsing field from response: {}".format(response))
                    
                paper["fields"] = final_fields

        with open("paper/step1/{conference_name}{year}.json".format(conference_name=self.conference_name, year=self.year), "w") as file:
            json.dump(papers, file, indent=4) 

def field_to_paper(conference_name, year, field_name):
    with open("fields/filtered_fields.json", "r") as file:
        fields_list_full = json.load(file)
    fields_list = [field["field"] for field in fields_list_full]
    
    for field in fields_list:
        if field_name.lower() in field.lower():
            field_name = field
            break

    with open("paper/classified/{conference_name}{year}.json".format(conference_name=conference_name, year=year), "r") as file:
        papers = json.load(file)
        selected_papers = [paper for paper in papers if field_name in paper.get("fields", [])]
    return selected_papers

if __name__ == "__main__":
    conference_name = "ICLR"
    year = 2025
    field_name = "Robotics"

    papers_in_field = field_to_paper(conference_name, year, field_name)

    print("Found {} papers in the field '{}' for conference {} in year {}.".format(len(papers_in_field), field_name, conference_name, year))

    a, b = random.sample(papers_in_field, 2)
    print("\nSample Paper 1:\nTitle: {}\nAbstract: {}\n".format(a["title"], a["abstract"]))
    print("\nSample Paper 2:\nTitle: {}\nAbstract: {}\n".format(b["title"], b["abstract"]))