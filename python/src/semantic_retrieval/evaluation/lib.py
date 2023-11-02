import logging
from typing import Callable, List, Sequence, Tuple

import pandas as pd
from semantic_retrieval.common import types
from semantic_retrieval.common.core import LOGGER_FMT

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class LocalFileSystemGenLLMEvalDataset(types.Record):
    names: Sequence[str]
    raw_retrieved_paths: Sequence[str]
    final_output_path: Sequence[str]

    @staticmethod
    def from_list(list_tuples: List[Tuple[str, str, str]]):
        names = []
        raw_retrieved_paths = []
        final_output_path = []
        for name, rrp, fop in list_tuples:
            names.append(name)
            raw_retrieved_paths.append(rrp)
            final_output_path.append(fop)

        return LocalFileSystemGenLLMEvalDataset(
            names=names, 
            raw_retrieved_paths=raw_retrieved_paths,
              final_output_path=final_output_path
            )

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



def file_contents(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        # todo deal with this
        logger.critical("exn=" + str(e))
        return ""



def local_filesystem_dataset_to_df(dataset: LocalFileSystemGenLLMEvalDataset):
    records = [
        {"name": name,
         "data_input": file_contents(rrp),
         "data_output": file_contents(fop)}
        for name, rrp, fop in zip(dataset.names, dataset.raw_retrieved_paths, dataset.final_output_path)
    ]
    return pd.DataFrame.from_records(records)


def run_correctness_checks(llm_eval_dataset: LocalFileSystemGenLLMEvalDataset):
    pass