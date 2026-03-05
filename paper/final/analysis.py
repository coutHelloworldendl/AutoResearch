import json
import os
import matplotlib.pyplot as plt

info = {
    "1.1": {
        "name": "Modular & Hierarchical Architectures",
        "items": [0, 2, 3, 7, 10, 12, 20, 22, 24, 26]
    },
    "1.2": {
        "name": "Multi-Agent & Collaborative Systems",
        "items": [4, 11, 12, 24]
    },
    "1.3": {
        "name": "Neuro-Symbolic & Hybrid Architectures",
        "items": [9, 27]
    },
    "1.4": {
        "name": "Specialized & Monolithic Models",
        "items": [5, 25, 29]
    },
    "2.1": {
        "name": "Explicit Planning & Tool-Augmented Reasoning",
        "items": [0, 2, 3, 16, 22, 26]
    },
    "2.2": {
        "name": "Implicit & Model-Based Reasoning",
        "items": [8, 19, 25]
    },
    "2.3": {
        "name": "Strategic & Game-Theoretic Reasoning",
        "items": [28]
    },
    "2.4": {
        "name": "Retrieval & Memory-Augmented Reasoning",
        "items": [15, 16]
    },
    "3.1": {
        "name": "Reinforcement & Imitation Learning",
        "items": [5, 12, 14, 17, 18, 23]
    },
    "3.2": {
        "name": "Adaptive & Continual Learning",
        "items": [2, 3, 15, 17, 20, 21]
    },
    "3.3": {
        "name": "Supervised & Meta-Learning",
        "items": [8, 13, 19]
    },
    "3.4": {
        "name": "Evolutionary & Curriculum Learning",
        "items": [14, 23]
    },
    "4.1": {
        "name": "Benchmark & Dataset Creation",
        "items": [1]
    },
    "4.2": {
        "name": "Perception & Modeling for Evaluation",
        "items": [1]
    },
    "5.1": {
        "name": "Multi-Agent Coordination",
        "items": [6, 12, 17]
    },
    "5.2": {
        "name": "Distributed & Online Optimization",
        "items": [6]
    },
    "6.1": {
        "name": "Web & GUI Interaction Agents",
        "items": [3, 10]
    },
    "6.2": {
        "name": "Conversational & Recommender Systems",
        "items": [29]
    },
    "6.3": {
        "name": "File Management & Tool Evolution",
        "items": [16]
    }
}

conferences = ["ACL", "CVPR", "EMNLP", "ICLR", "NIPS"]
years = [2023, 2024, 2025]

INPUT_PATH = "/home/squirrel/workspace/AutoResearch/paper/step1/{conference}{year}.json"
KEYWORD_PATH = "/home/squirrel/workspace/AutoResearch/paper/step7/Agents.json"

with open(KEYWORD_PATH, "r") as f:
    KEYWORDS = json.load(f)


def paper_of_the_year(year):
    sum = 0
    for conf in conferences:
        path = INPUT_PATH.format(conference=conf, year=year)
        with open(path, "r") as f:
            data = json.load(f)
            sum += len(data)
    return sum

def trend_analysis(paper_list):
    trend = {}
    for paper in paper_list:
        conference_and_year = paper.split("-")[0]
        year = int(conference_and_year[-4:])

        trend[year] = trend.get(year, 0) + 1

    return trend

if __name__ == "__main__":
    for key, value in info.items():
        paper_list = []
        for idx in value["items"]:
            cluster = KEYWORDS[idx]
            paper_list.extend(cluster["paper_ids"])
        
        trend = trend_analysis(paper_list)
        years = sorted(trend.keys())
        counts = [trend[year] for year in years]  
        plt.plot(years, counts, marker='o', label=value["name"])
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.title(value["name"])
        plt.xticks(years)
        plt.grid()
        plt.legend()
        plt.savefig(f'/home/squirrel/workspace/AutoResearch/paper/final/fig/{key}.png')
        plt.clf()