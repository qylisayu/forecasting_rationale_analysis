import json
from typing import List
from nlparam import run_classification


if __name__ == '__main__':
    data_path = '../data_processing/rationale_analysis_nlparam_classification_input_data_train.json'
    with open(data_path, "r") as f:
        task_dict = json.load(f)

    texts: List[str] = task_dict["texts"]
    labels: List[int] = task_dict["labels"]
    
    K: int = 3
    goal: str = "Here are some forecasters' reationales for their predictions. I want to understand what features are important for accurate forecasts."
    
    # runnning the classification model
    clf_result = run_classification(texts, labels, K, goal)

    w = clf_result["w"]
    predicates = clf_result["predicates"]
    for i, predicate in enumerate(predicates):
        print(f"{predicate}: {w[0][i]}")
