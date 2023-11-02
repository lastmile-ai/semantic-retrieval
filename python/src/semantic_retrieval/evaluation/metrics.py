from sklearn.metrics import accuracy_score

from semantic_retrieval.evaluation.lib import SampleEvalDataset


def accuracy_metric(dataset: SampleEvalDataset) -> float:
    return accuracy_score(dataset.ground_truth, dataset.output)  # type: ignore [fixme]
