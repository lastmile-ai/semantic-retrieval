import asyncio
import logging
import sys
from typing import List, Tuple

import pandas as pd

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.evaluation.lib import (
    SampleEvalDataset,
    evaluate_sample_local_filesystem,
)
from semantic_retrieval.evaluation.metrics import accuracy_metric

from semantic_retrieval.examples.financial_report.config import (
    Config,
    get_config,
    resolve_path,
    set_up_script,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


async def main(argv: List[str]):
    loggers = [logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_evaluate_report(config)


import re


def parse_raw_re(s_out: str) -> List[Tuple[str, str, str, str, str]]:
    try:
        return re.findall(
            r"([A-Z]+).*([ \$])([\d,\.]+)( *)(million|billion)?",
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
        ticker, _, number, _, units = row
        number_parsed = float(number.replace(",", ""))
        convert_factor = 1

        if units.lower().startswith("b"):
            convert_factor = 1000

        out.append(
            {"ticker": ticker, "value_millions": int(number_parsed * convert_factor)}
        )
    return pd.DataFrame.from_records(out)


def file_contents(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        # todo deal with this
        logger.critical("exn=" + str(e))
        return ""


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
    accuracy_pct = 100 * evaluate_sample_local_filesystem(
        path_output,
        path_ground_truth,
        path_muncher,
        accuracy_metric,
    )

    print(f"Accuracy: {accuracy_pct:.2f}%")


if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
