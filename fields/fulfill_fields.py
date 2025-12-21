import json
from tqdm import tqdm
from api import API

with open("fields/fields.json", "r") as f:
    fields_data = json.load(f)

api = API(model_name="deepseek")
focus_prompt = \
'''
Please provide an about 30 to 50 words explanation of the field {field}, including what {field} is focused on, also list some key indicators which are commonly used in this field.
Output Example:
Robotics Definition: Research involving hardware systems equipped with input sensors and mechanical kinematics capable of producing joint movements. These systems are controlled by learning-based algorithms that facilitate automatic or robust mappings from sensory inputs to actuator outputs.
Key Indicators:
- Reinforcement Learning in robotic contexts
- Imitation Learning for physical systems
'''

perspective_prompt = \
'''
Please provide 5 very core perspectives when reading papers in the field of {field}. Each perspective should be concise, about 5 to 10 words. The output should be in a list format.
There maybe some views to help you understand what perspectives are:{rdr_perspective_example}.
Output Example(for Machine Learning):
[Datasets/Features, Learning Algorithm (e.g., SVM, Decision Tree), Predictive Model/Function, Loss Function Minimization, Training/Validation Pipeline]
'''

for field in tqdm(fields_data):
    filled_prompt = focus_prompt.format(field=field["domain"])
    explanation = api.forward(model_name="deepseek", prompt=filled_prompt)
    field["explanation"] = explanation

for field in tqdm(fields_data):
    filled_prompt = perspective_prompt.format(field=field["domain"], rdr_perspective_example=field.get("rdr_perspective_example", ""))
    perspectives = api.forward(model_name="deepseek", prompt=filled_prompt)
    field["perspectives"] = perspectives

for field in fields_data:
    field["field"] = field.pop("domain")
    if "rdr_perspective_example" in field:
        field.pop("rdr_perspective_example")
    if "core_focus" in field:
        field.pop("core_focus")

for field in fields_data:
    perspectives_str = field["perspectives"].replace("[", "").replace("]", "").replace("\n", " ").replace(", ",",").replace(" ,",",").replace("  ", " ")
    try:
        field_list = perspectives_str.split(",")
        field["perspectives"] = [persp.strip() for persp in field_list if persp.strip()]
    except:
        raise KeyError(f"Field {field['field']} has invalid perspectives format.")

with open("fields/filtered_fields.json", "w") as f:
    json.dump(fields_data, f, indent=4)