from typing import TypeVar
from sklearn.metrics import accuracy_score

from semantic_retrieval.evaluation.lib import EvalSetPair, SampleEvalDataset

T = TypeVar("T")

def accuracy_metric(dataset: SampleEvalDataset) -> float:
    return accuracy_score(dataset.ground_truth, dataset.output)  # type: ignore [fixme]


def jaccard_similarity(dataset: EvalSetPair[T]) -> float:
    intersection = dataset.input_set & dataset.output_set
    # print(f"{dataset=}")
    union = dataset.input_set | dataset.output_set
    # print(f"{intersection=}, {union=}")

    diff = dataset.input_set - dataset.output_set, dataset.output_set - dataset.input_set
    # print(f"missing={diff}")
    return len(intersection) / len(union)