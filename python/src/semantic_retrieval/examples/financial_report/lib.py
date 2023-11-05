from functools import partial
import os
import re
import json
import logging
from typing import Any
import pandas as pd
from result import Ok, Result
from semantic_retrieval.access_control.access_function import always_allow
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity
from semantic_retrieval.common.core import LOGGER_FMT, file_contents
from semantic_retrieval.evaluation.lib import (
    IDSet,
    IDSetPairEvalDataset,
    SampleEvaluationParams,
)

from semantic_retrieval.examples.financial_report.financial_report_document_retriever import (
    PortfolioData,
)
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever

from semantic_retrieval.evaluation import metrics


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


def portfolio_df_to_dict(df: pd.DataFrame) -> PortfolioData:
    return PortfolioData(
        df.set_index("Company")
        .astype(float)
        .fillna(0)
        .query("Shares > 0")["Shares"]
        .to_dict()
    )


def completion_model_output_portfolio_data_muncher(path: str) -> IDSet:
    contents = file_contents(path)
    logger.debug(f"{path=}")
    parsed = re.findall(
        r"([A-Z]+)['\)].*\n",
        contents,
        flags=re.IGNORECASE,
    )
    tickers: IDSet = IDSet(set(parsed))
    return tickers


async def test_1_2_data_muncher(
    path_input: str, path_output: str
) -> Result[IDSetPairEvalDataset, str]:
    logger.debug(f"{path_input=}, {path_output=}")

    def raw_retrieved_chunks_data_muncher(path: str) -> IDSet:
        contents = file_contents(path)
        obj = json.loads(contents)
        out: IDSet = IDSet({obj_["company"] for obj_ in obj})
        return out

    input_set = raw_retrieved_chunks_data_muncher(path_input)
    output_set = completion_model_output_portfolio_data_muncher(path_output)

    return Ok(IDSetPairEvalDataset(input_set=input_set, output_set=output_set))


async def test_3_4_data_muncher(
    path_input: str, path_output: str
) -> Result[IDSetPairEvalDataset, str]:
    logger.debug(f"{path_input=}, {path_output=}")

    def _key_set(df: pd.DataFrame) -> IDSet:
        dict_portfolio = portfolio_df_to_dict(df)
        idset: IDSet = IDSet(set(dict_portfolio.keys()))
        return idset

    async def portfolio_data_muncher(path: str) -> Result[IDSet, str]:
        portfolio = await CSVRetriever(
            path, AuthenticatedIdentity.mock(), always_allow()
        ).retrieve_data()

        return portfolio.map(_key_set)

    res_portfolio_set = await portfolio_data_muncher(path_input)
    output_set = completion_model_output_portfolio_data_muncher(path_output)

    def _portfolio_set_to_set_pair(portfolio_set: IDSet) -> IDSetPairEvalDataset:
        return IDSetPairEvalDataset(input_set=portfolio_set, output_set=output_set)

    return res_portfolio_set.map(_portfolio_set_to_set_pair)


async def test_case_to_sample_eval_params(
    test_case: Any, root_dir: str
) -> Result[SampleEvaluationParams[IDSet], str]:
    """
    Helper function to wrap a test case in the required structure
    for `evaluate()`.

    Specifically, return a SampleEvaluationParams
    containing the output sample (returned ID set)
    and the evaluation function.

    In this case, the evaluation function is a closure
    which compares the output ID set to the
    (ground-truth) reference set.

    The comparison is done using the Jaccard similarity,
    leveraging the `metrics` library.
    """

    name, muncher, input_path, output_path = test_case
    input_path = os.path.join(root_dir, input_path)
    output_path = os.path.join(root_dir, output_path)
    logger.info(f"\n\nPreparing test case {name}:\n{input_path=}\n{output_path=}")
    idset_pair: Result[IDSetPairEvalDataset, str] = await muncher(
        input_path, output_path
    )
    logger.debug(f"{name=}")
    logger.debug(f"{idset_pair=}")

    # Eval params contains the data and the evaluation function.
    # In this case, the data is the output ID set,
    # and the evaluation function is the closure that compares
    # that set to the reference set.
    eval_params = idset_pair.map(
        partial(metrics.id_set_pair_to_jaccard_params, name=name)
    )

    logger.info("\n\nCreated evaluation params for test case:\n\n")
    logger.info(eval_params.map_or("err", str))

    return eval_params
