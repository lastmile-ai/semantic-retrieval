import asyncio
import logging
import sys
from typing import List


import pandas as pd

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.evaluation.lib import (
    eval_res_to_df,
    evaluate,
)

from semantic_retrieval.examples.financial_report.config import (
    Config,
    get_config,
    set_up_script,
)


import pandas as pd
from semantic_retrieval.examples.financial_report.eval_test_config import TEST_CASES

from semantic_retrieval.examples.financial_report import lib as fr_lib
from semantic_retrieval.evaluation import lib as evaluation_lib

from semantic_retrieval.evaluation import metrics
from semantic_retrieval.functional.functional import result_reduce_list_separate


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

ROOT_DATA_DIR = "examples/example_data/financial_report/"


async def run_evaluate_report(config: Config):
    evaluation_params_list = [
        await fr_lib.test_case_to_sample_eval_params(tc, ROOT_DATA_DIR)
        for tc in TEST_CASES
    ]

    # Helper function to pretty print all the results
    def _print(df: pd.DataFrame) -> None:
        return print(df.set_index(["name", "interpretation"]))

    # Separate out the valid test params from the error ones.
    evaluation_params_valid, evaluation_params_errs = result_reduce_list_separate(
        evaluation_params_list
    )

    print(f"Evaluation params with errors: {evaluation_params_errs}")

    # Run `evaluate()` on the valid params.
    eval_res = evaluate(evaluation_params_valid)

    # Print results
    df_eval_res = eval_res.map(eval_res_to_df)
    df_eval_res.map(_print)


async def main(argv: List[str]):  # type: ignore
    loggers = [logger, metrics.logger, fr_lib.logger, evaluation_lib.logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))
    return await run_evaluate_report(config)


if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
