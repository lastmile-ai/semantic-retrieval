import logging
from typing import Callable, Generic, List, Sequence, Set, Tuple, TypeVar

import pandas as pd
from semantic_retrieval.common import types
from semantic_retrieval.common.core import LOGGER_FMT

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

class SampleEvalDataset(types.Record):
    output: Sequence[float | int]
    ground_truth: Sequence[float | int]


T = TypeVar("T")

class EvalSetPair(types.Record, Generic[T]):
    input_set: Set[T]
    output_set: Set[T]  



# TODO: newtype
Metric = Callable[[SampleEvalDataset], float]


PathMuncher = Callable[[str, str], EvalSetPair[str]]

def evaluate_sample_local_filesystem(
    path_output: str,
    path_ground_truth: str,
    path_muncher: PathMuncher,
    metric: Metric,
):
    return 1



T = TypeVar("T")


class LocalFileSystemGenLLMEvalDataset(types.Record):
    names: Sequence[str]
    input_paths: Sequence[str]
    final_output_path: Sequence[str]

    @staticmethod
    def from_list(list_tuples: Sequence[Tuple[str, str, str]]):
        names = []
        input_paths = []
        final_output_path = []
        for name, rrp, fop in list_tuples:
            names.append(name)
            input_paths.append(rrp)
            final_output_path.append(fop)

        return LocalFileSystemGenLLMEvalDataset(
            names=names, 
            input_paths=input_paths,
              final_output_path=final_output_path
            )



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
        for name, rrp, fop in zip(dataset.names, dataset.input_paths, dataset.final_output_path)
    ]
    return pd.DataFrame.from_records(records)



async def evaluate_llm_eval_dataset_local_filesystem(
    llm_eval_dataset: LocalFileSystemGenLLMEvalDataset,
    data_munchers: List[PathMuncher],
    metrics: List[Metric]
):
  results = []
  for name, ip, op, dm, m in zip(llm_eval_dataset.names, llm_eval_dataset.input_paths, llm_eval_dataset.final_output_path, data_munchers, metrics):
      munched = await dm(ip, op)
      value = m(munched)
      results.append(
          {
              "name": name,
              "value": value,
          }
      )          

  return pd.DataFrame.from_records(results)