import logging
from typing import Awaitable, Callable, Sequence, Set, Tuple, TypeVar

import pandas as pd
from semantic_retrieval.common import types
from semantic_retrieval.common.core import LOGGER_FMT


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

T = TypeVar("T")


class IDSetPairEvalDataset(types.Record):
    input_set: Set[str]
    output_set: Set[str]

class NumericalEvalDataset(types.Record):
    output: Sequence[float | int]
    ground_truth: Sequence[float | int]


IDSetPairEvalDataPathMuncher = Callable[[str, str], Awaitable[IDSetPairEvalDataset]]
NumericalEvalDataPathMuncher = Callable[[str, str], Awaitable[NumericalEvalDataset]]

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


class LocalFileSystemIDSetPairEvalDatasetConfig(types.Record):
    """
    Container struct for the logic you need to run eval
    on a local dataset, where the structured data is a pair of ID sets
    """
    fn_path_muncher: IDSetPairEvalDataPathMuncher
    metric: Callable[[IDSetPairEvalDataset], float]


class LocalFileSystemNumericalEvalDatasetConfig(types.Record):
    """
    Container struct for the logic you need to run eval
    on a local dataset, where the structured data is a
    pair of numerical arrays
    """
    fn_path_muncher: NumericalEvalDataPathMuncher
    metric: Callable[[NumericalEvalDataset], float]


LocalFileSystemEvalDatasetConfig = LocalFileSystemIDSetPairEvalDatasetConfig | LocalFileSystemNumericalEvalDatasetConfig


async def evaluate_sample_local_filesystem(
    path_output: str,
    path_ground_truth: str,
    local_filesystem_eval_dataset_config: LocalFileSystemEvalDatasetConfig
) -> float:
    match local_filesystem_eval_dataset_config:
        # Code happens to be the same in the two branches.
        # These types should actually be one generic type,
        # But that's not well-supported, e.g. bounded generics.
        case LocalFileSystemIDSetPairEvalDatasetConfig(metric=m, fn_path_muncher=pm):
            dataset_for_sample = await pm(path_ground_truth, path_output)
            return m(dataset_for_sample)
        case LocalFileSystemNumericalEvalDatasetConfig(metric=m, fn_path_muncher=pm):
            dataset_for_sample = await pm(path_ground_truth, path_output)            
            return m(dataset_for_sample)



def file_contents(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        # TODO [P1] deal with this
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
    eval_configs: Sequence[LocalFileSystemEvalDatasetConfig]
):
  results = []
  for name, ip, op, cfg in zip(llm_eval_dataset.names, llm_eval_dataset.input_paths, llm_eval_dataset.final_output_path, eval_configs):
      value = await evaluate_sample_local_filesystem(op, ip, cfg)
      results.append(
          {
              "name": name,
              "value": value,
          }
      )          

  return pd.DataFrame.from_records(results)