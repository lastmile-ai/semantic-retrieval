from typing import Callable, Sequence
from semantic_retrieval.common import types


class SampleEvalDataset(types.Record):
    output: Sequence[float | int]
    ground_truth: Sequence[float | int]


# TODO: newtype
Metric = Callable[[SampleEvalDataset], float]


def evaluate_sample_local_filesystem(
    path_output: str,
    path_ground_truth: str,
    path_muncher: Callable[[str, str], SampleEvalDataset],
    metric: Metric,
) -> float:
    dataset_for_sample = path_muncher(path_output, path_ground_truth)
    value = metric(dataset_for_sample)
    return value
