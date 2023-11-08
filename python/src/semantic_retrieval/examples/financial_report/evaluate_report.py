import asyncio
import logging
import sys
from typing import List


from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.evaluation.lib import (
    eval_res_to_df,
    evaluate,
)

from semantic_retrieval.examples.financial_report.lib import config

from semantic_retrieval.evaluation import lib as evaluation_lib

from semantic_retrieval.evaluation import metrics
from semantic_retrieval.examples.financial_report.lib.eval import (
    get_test_suite,
    test_case_to_sample_eval_params,
)


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

ROOT_DATA_DIR = "examples/example_data/financial_report/"


async def run_evaluate_report(config_instance: config.Config):
    # Convert the raw test cases to evaluation params.
    # See lib_eval/test_case_to_sample_eval_params for more details.
    evaluation_params_list = [
        await test_case_to_sample_eval_params(tc, ROOT_DATA_DIR) for tc in get_test_suite()
    ]

    # Separate out the valid test params from the error ones.
    evaluation_params_valid = [ep.unwrap() for ep in evaluation_params_list if ep.is_ok()]

    logger.info("Evaluating the following test cases:")
    for eval_param in evaluation_params_valid:
        logger.info(f"\n\nEval params: {eval_param}")

    eval_res = evaluate(evaluation_params_valid)

    if eval_res.is_err():
        logger.critical(f"Error evaluating: {eval_res.err()}")

    df_eval_res = eval_res_to_df(eval_res.unwrap())

    print(df_eval_res.set_index(["name", "interpretation"]))


async def main(argv: List[str]):
    loggers = [logger, metrics.logger, evaluation_lib.logger]

    args = config.set_up_script(argv, loggers)
    config_instance = config.get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))
    return await run_evaluate_report(config_instance)


if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
