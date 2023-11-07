# from typing import Any, Dict, TypeVar
import argparse
import logging
import os
from typing import List, Sequence

from dotenv import load_dotenv
from semantic_retrieval.common.types import Record
from semantic_retrieval.utils.configs.configs import argparsify
from semantic_retrieval.utils.logging import set_log_level


class Config(Record):
    # Keys
    pinecone_key: str
    openai_key: str

    # Paths are relative to `data_root` unless specified as "_abs"
    # DO NOT UPSERT
    # Remote paths / IDs
    namespace: str = "ea4bcf44-e0f3-46ff-bf66-5b1f9e7502df"
    index_name: str = "examples"
    pinecone_environment: str = "asia-southeast1-gcp-free"

    # Local paths
    # TODO [P1]: double check this: data_root is relative to cwd.
    data_root: str = "examples/example_data/financial_report"
    path_10ks: str = "10ks"
    metadata_db_name: str = "the_metadata_db"
    portfolio_csv_dir: str = "portfolios"

    # Misc
    viewer_role: str = "advisor/jonathan"
    client_name: str = "sarmad"
    top_k: int = 10
    overfetch_factor: float = 20.0

    # make sure this is correct!
    chunk_size_limit: int = 500

    # assume 8k (GPT4) and leave room for the instruction and
    # generated output
    retrieved_context_limit: int = 4000
    retrieval_query: str = "covid 19 impact"
    # structure_prompt: str = "Numbered list, one security per list item,"
    # data_extraction_prompt: str = "data_extraction_prompt"

    log_level: str = "WARNING"

    # Eval
    sample_output_path: str = "portfolio_10k_net_income_report.txt"
    ticker_eval_ground_truth_path: str = "ticker_numerical_eval_gt.csv"


def get_config(args: argparse.Namespace):
    args_dict = dict(
        openai_key=os.getenv("OPENAI_API_KEY"),
        pinecone_key=os.getenv("PINECONE_API_KEY"),
    )
    for k, v in vars(args).items():
        if v is not None:
            args_dict[k] = v

    return Config(**args_dict)  # type: ignore


def resolve_path(data_root: str, path: str) -> str:
    """
    data_root is relative to csv
    path is relative to data_root
    """

    return os.path.join(os.getcwd(), data_root, path)


def set_up_script(argv: Sequence[str], loggers: List[logging.Logger]):
    load_dotenv()

    parser = argparsify(Config)
    args = parser.parse_args(argv[1:])

    config = get_config(args)

    set_log_level(config.log_level, loggers)

    return args


def get_metadata_db_path(config: Config):
    file = f"{config.metadata_db_name}_{config.namespace}_{config.index_name}_{config.pinecone_environment}.json"
    path = os.path.join("artifacts", file)
    return resolve_path(config.data_root, path)
