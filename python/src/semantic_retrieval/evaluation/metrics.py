import numpy as np

from semantic_retrieval.evaluation.lib import SampleEvalDataset


def accuracy_metric(dataset: SampleEvalDataset):
    preds = np.array(dataset.output)
    ground_truth = np.array(dataset.ground_truth)
    return (preds == ground_truth).astype(int).mean()
