import numpy as np 
import pandas as pd 
import json

corpus = pd.read_csv("../data/corpus.csv")
ques = pd.read_csv("../data/question.csv")

with open("../data/ground_truth.json", "r") as f:
    gt = json.load(f)
    new_gt = []
    for item in gt:
        for k, v in item.items():
            new_gt.append({
                "qid": k,
                "cids": v
            })
    gt = new_gt

dataset = []
for item in gt:
    qid = item['qid']
    question = ques[ques['qid'] == int(qid)]['question'].to_list()[0]
    dataset.append((question, item['cids']))

test_size = int(0.1 * len(dataset))

test_dataset, train_dataset = dataset[:test_size], dataset[test_size:]