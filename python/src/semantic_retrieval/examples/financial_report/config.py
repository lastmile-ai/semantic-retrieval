# from typing import Any, Dict, TypeVar
import argparse
import json
import logging
import os
from typing import List, Type

from dotenv import load_dotenv
from semantic_retrieval.common.types import Record
from semantic_retrieval.utils.configs.configs import combine_dicts, remove_nones


class Config(Record):
    # Keys
    pinecone_key: str
    openai_key: str

    # Paths are relative to `data_root` unless specified as "_abs"
    # DO NOT UPSERT
    # Remote paths / IDs
    namespace: str = "ea4bcf44-e0f3-46ff-bf66-5b1f9e7502df"
    index_name: str = "test-financial-report"
    pinecone_environment: str = "asia-southeast1-gcp-free"

    # Local paths
    # TODO: double check this: data_root is relative to cwd.
    data_root: str = "examples/example_data/financial_report"
    metadata_db_path: str = "artifacts/metadata_db_py_v2.json"
    portfolio_csv_dir: str = "portfolios"

    # Misc
    client_name: str = "sarmad"
    top_k: int = 10
    overfetch_factor: float = 5.0

    # make sure this is correct!
    chunk_size_limit: int = 500

    # assume 8k (GPT4) and leave room for the instruction and
    # generated output
    retrieved_context_limit: int = 4000
    retrieval_query: str = "overall cash flow"
    structure_prompt: str = "Numbered list"
    data_extraction_prompt: str = "data_extraction_prompt"

    log_level: str = "WARNING"

    # Eval
    sample_output_path: str = "portfolio_10k_net_income_report.txt"
    ticker_eval_ground_truth_path: str = "ticker_numerical_eval_gt.csv"

    def __repr__(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()


def add_parser_argument(parser, field_name, field):  # type: ignore
    field_name = field_name.replace("_", "-")
    the_type = field.annotation
    parser.add_argument(f"--{field_name}", type=the_type)


def add_parser_arguments(parser, fields):  # type: ignore
    for field_name, field in fields.items():
        add_parser_argument(parser, field_name, field)


def argparsify(r: Record | Type[Record]):
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser, r.model_fields)
    return parser


def get_config(args: argparse.Namespace):
    # TODO combine stuff cleaner
    args_resolved = combine_dicts(
        [
            remove_nones(d)
            for d in [
                vars(args),
                dict(
                    openai_key=os.getenv("OPENAI_API_KEY"),
                    pinecone_key=os.getenv("PINECONE_API_KEY"),
                ),
            ]
        ]
    )
    return Config(**args_resolved)


def set_log_level(log_level: int | str, loggers: List[logging.Logger]):
    ll: int = -1
    match log_level:
        case int():
            ll = int(log_level)
        case str():
            ll = getattr(logging, log_level.upper())

    for logger_ in loggers:
        logger_.setLevel(ll)


def resolve_path(data_root: str, path: str) -> str:
    """
    data_root is relative to csv
    path is relative to data_root
    """

    return os.path.join(os.getcwd(), data_root, path)
