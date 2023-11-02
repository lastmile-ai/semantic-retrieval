import asyncio
import json
import logging
import os
import sys
import re

from typing import Callable, List, Tuple
from dotenv import load_dotenv

import pandas as pd

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.evaluation.lib import (
    EvalSetPair,
    LocalFileSystemGenLLMEvalDataset,
    SampleEvalDataset,
    evaluate_sample_local_filesystem,
    file_contents,
)
from semantic_retrieval.evaluation.metrics import accuracy_metric, jaccard_similarity

from semantic_retrieval.examples.financial_report.config import (
    Config,
    argparsify,
    get_config,
    resolve_path,
    set_log_level,
)

import sys
import pandas as pd

from semantic_retrieval.examples.financial_report.evaluate_report import file_contents
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever
from semantic_retrieval.common import types
from semantic_retrieval.evaluation.lib import (
    LocalFileSystemGenLLMEvalDataset,
    SampleEvalDataset,
    local_filesystem_dataset_to_df,
    Metric
)

import glob
import os


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


async def main(argv: List[str]):
    loggers = [logger]

    args = _set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    if args.use_v2:
        return await run_evaluate_report_v2()
    else:
        return await run_evaluate_report(config)


def _set_up_script(argv, loggers):
    load_dotenv()

    parser = argparsify(Config)
    parser.add_argument("--use-v2", action="store_true")
    args = parser.parse_args(argv[1:])

    config = get_config(args)

    set_log_level(config.log_level, loggers)

    return args


def parse_raw_re(s_out: str) -> List[Tuple[str, str, str, str, str]]:
    try:
        return re.findall(
            # r"([A-Z]+)[.*]*([ \$])([\d,\.]+)( *)(million|billion)?",
            r"([A-Z]+)['\)].*\n",
            s_out,
            flags=re.IGNORECASE,
        )
    except Exception as e:
        # todo deal with this
        logger.warning("parse output exn, returning empty list, exn=", str(e))
        return []


def parse_re_output_to_df(re_output: List[Tuple[str, str, str, str, str]]):
    out = []
    for row in re_output:
        # ticker, _, number, _, units = row
        ticker = row
        out.append({"ticker": ticker})
        continue
        number_parsed = float(number.replace(",", ""))
        convert_factor = 1

        if units.lower().startswith("b"):
            convert_factor = 1000

        out.append(
            {"ticker": ticker, "value_raw": number, "value_millions": int(number_parsed * convert_factor)}
        )
    return pd.DataFrame.from_records(out)


def gen_output_to_df(s_out: str) -> pd.DataFrame:
    return parse_re_output_to_df(parse_raw_re(s_out))


def path_muncher(output_path: str, ground_truth_path: str) -> SampleEvalDataset:
    logger.debug("HERE" + output_path + "\n" + ground_truth_path)
    df_gt = pd.read_csv(ground_truth_path)
    logger.debug(f"GT={df_gt}")

    s_out = file_contents(output_path)
    re_output = parse_raw_re(s_out)
    df = parse_re_output_to_df(re_output)

    df_join = df.set_index("ticker").join(
        df_gt.set_index("ticker"), how="right", lsuffix="_output", rsuffix="_gt"
    )

    logger.info(f"Outputs and Ground truth:\n{df_join}")

    # TODO this better
    return SampleEvalDataset(
        output=df_join.value_millions.tolist(),
        ground_truth=df_join.net_earnings_millions_2022.tolist(),
    )


async def run_evaluate_report(
    config: Config,
):
    path_output = resolve_path(config.data_root, config.sample_output_path)

    path_ground_truth = resolve_path(
        config.data_root, config.ticker_eval_ground_truth_path
    )
    logger.info(f"Starting evaluation\n{path_ground_truth=}\n{path_output=}")
    accuracy_pct = 100 * evaluate_sample_local_filesystem(
        path_output,
        path_ground_truth,
        path_muncher,
        accuracy_metric,
    )

    print(f"Accuracy: {accuracy_pct:.2f}%")






# TODO
# def raw_retrieved_chunks_data_muncher_values(
#     path: str
# ):
#     contents = file_contents(path)
#     # logger.debug(f"{contents=}")
#     obj = json.loads(contents)
#     return {obj_['company'] for obj_ in obj}
    


async def test_1_2_data_muncher(
    contents_input: str,
    contents_output: str    
):    
    input_set = raw_retrieved_chunks_data_muncher(contents_input)
    output_set = completion_model_output_portfolio_data_muncher(contents_output)
    return EvalSetPair(input_set=input_set, output_set=output_set)

async def test_3_4_data_muncher(
    contents_input: str,
    contents_output: str    
):
    portfolio_set = await portfolio_data_muncher(contents_input)
    output_set = completion_model_output_portfolio_data_muncher(contents_output) 

    return EvalSetPair(input_set=portfolio_set, output_set=output_set)

async def run_evaluate_report_v2():
    ROOT_DATA_DIR = "../../../../../examples/example_data/financial_report/"
    ROOT_DATA_DIR = "examples/example_data/financial_report"

    ARTIFACTS = os.path.join(ROOT_DATA_DIR, "artifacts")
   
    TEST_CASES = [
    (name, os.path.join(ARTIFACTS, ip), os.path.join(ARTIFACTS, op))
    for name, ip, op in [
        ("net_income_vs_retrieved", "raw_retrieved_chunks_10k_net_income.json", "portfolio_10k_net_income_report.txt"),
        ("covid_vs_retrieved", "raw_retrieved_chunks_10k_covid.json", "portfolio_10k_covid_report.txt"),
    ]
] + [
    (name, os.path.join(ROOT_DATA_DIR, ip), os.path.join(ARTIFACTS, op))
    for name, ip, op in [
        ("net_income_e2e", "portfolios/sarmad_portfolio.csv", "portfolio_10k_net_income_report.txt"),
        ("covid_e2e", "portfolios/sarmad_portfolio.csv", "portfolio_10k_covid_report.txt"),                    
    ]
]    

    llm_eval_dataset = LocalFileSystemGenLLMEvalDataset.from_list(TEST_CASES)
    data_munchers = [test_1_2_data_muncher] * 2 + [test_3_4_data_muncher] * 2
    metrics = [jaccard_similarity] * 4
    return await evaluate_llm_eval_dataset_local_filesystem(llm_eval_dataset, data_munchers, metrics)

if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
