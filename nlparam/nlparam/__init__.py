from pathlib import Path
from .nlparam_logging import logger
import numpy as np

# Get the current file's directory.
current_directory = Path(__file__)
TEMPLATE_DIRECTORY = current_directory.parent / "templates"
DATA_DIR = current_directory.parent.parent / "data"
DEFAULT_VALIDATOR_NAME = "gpt-4o-mini"
# DEFAULT_EMBEDDER_NAME = "hkunlp/instructor-xl"
DEFAULT_EMBEDDER_NAME = "hkunlp/instructor-base"
DEFAULT_PROPOSER_NAME = "gpt-4o"
EXPERIMENT_VALIDATOR_NAME = "google/flan-t5-xl"
EXPERIMENT_PROPOSER_NAME = "gpt-3.5-turbo-0613"
DEFAULT_LLM_CONFIG = {
    "validator": {"model_name": DEFAULT_VALIDATOR_NAME},
    "proposer": {"model_name": DEFAULT_PROPOSER_NAME},
    "embedder": {"model_name": DEFAULT_EMBEDDER_NAME},
}

from .llm.query import query_wrapper
from .llm.propose import Proposer
from .llm.validate import get_validator_by_name, validate_descriptions, Validator
from .llm.embedder import Embedder
from .models.cluster_model import ClusteringModel
from .models.classification_model import ClassificationModel
from .models.timeseries_model import TimeSeriesModel
from .models.abstract_model import ModelOutput
from typing import List, Dict
from sklearn.linear_model import LogisticRegression


def continuous_arr(arr, lr=0.005):
    init = np.mean(arr[:50])
    cur = init
    continuous = [cur]
    for i in range(1, len(arr)):
        cur = (1-lr) * cur + lr * arr[i]
        continuous.append(cur)
    return continuous


# def dedup_text_labels(texts, labels):
#     text2label = {text: label for text, label in zip(texts, labels)}
#     texts = [text for text in text2label]
#     labels = [text2label[text] for text in texts]
#     return texts, labels

def dedup_text_labels(texts, labels, scores):
    # Create a mapping for texts to their labels and scores
    text2info = {text: (label, score) for text, label, score in zip(texts, labels, scores)}

    # Extract deduplicated texts, labels, and scores
    texts = [text for text in text2info]
    labels = [text2info[text][0] for text in texts]
    scores = [text2info[text][1] for text in texts]

    return texts, labels, scores

def run_classification(texts: List[str], labels: List[int], scores: List[float], 
                       val_texts: List[str], val_labels: List[int], val_scores: List[float],
                       test_texts: List[str], test_labels: List[int], test_scores: List[float],
                       K: int, goal: str, num_iterations: int, lr: float) -> ModelOutput:
    texts, labels, scores = dedup_text_labels(texts, labels, scores)
    val_texts, val_labels, val_scores = dedup_text_labels(val_texts, val_labels, val_scores)
    test_texts, test_labels, test_scores = dedup_text_labels(test_texts, test_labels, test_scores)
    model = ClassificationModel(
        texts=texts,
        labels=labels,
        scores=scores,
        val_texts=val_texts,
        val_labels=val_labels,
        val_scores=val_scores,
        K=K,
        goal=goal,
        num_iterations=num_iterations,
        lr = lr
    )
    result: ModelOutput = model.fit()
    predicate2text2matching: Dict[str, Dict[str, int]] = result.predicate2text2matching

    predicates = list(predicate2text2matching.keys())
    features = [
        [predicate2text2matching[predicate][text] for predicate in predicates]
        for text in texts
    ]
    clf = LogisticRegression()
    clf.fit(features, labels)
    w = clf.coef_
    
    return {
        "predicate2text2matching": predicate2text2matching,
        "w": w,
        "predicates": predicates,
    }


def dedup_texts(texts_by_time: List[str]) -> List[str]:
    texts_by_time_ = []
    for text in texts_by_time:
        if text not in texts_by_time_:
            texts_by_time_.append(text)
    return texts_by_time_


def run_time_series(
    texts_by_time: List[str],
    K: int,
    goal: str,
    dummy: bool = False,
) -> ModelOutput:

    texts_by_time_ = dedup_texts(texts_by_time)

    model = TimeSeriesModel(
        texts_by_time=texts_by_time_,
        K=K,
        goal=goal,
        dummy=dummy,
    )
    result: ModelOutput = model.fit()
    predicate2text2matching: Dict[str, Dict[str, int]] = result.predicate2text2matching

    predicates = list(predicate2text2matching.keys())
    features = [
        [predicate2text2matching[predicate][text] for predicate in predicates]
        for text in texts_by_time
    ]

    continuous_arrs = [continuous_arr(arr) for arr in np.array(features).T]

    return {
        "predicate2text2matching": predicate2text2matching,
        "predicates": predicates,
        "features": features,
        "curves": continuous_arrs
    }


def run_clustering(texts: List[str], K: int, goal: str) -> ModelOutput:
    texts = list(set(texts))
    model = ClusteringModel(
        texts=texts,
        K=K,
        goal=goal,
    )

    result: ModelOutput = model.fit()
    predicate2text2matching: Dict[str, Dict[str, int]] = result.predicate2text2matching
    predicates = list(predicate2text2matching.keys())

    return {
        "predicate2text2matching": predicate2text2matching,
        "predicates": predicates,
    }
