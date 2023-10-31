# from typing import Any, Dict, TypeVar
import argparse
from typing import Type
from semantic_retrieval.common.types import Record


class Config(Record):
    # Keys
    pinecone_key: str
    openai_key: str

    # Paths are relative to `data_root` unless specified as "_abs"
    # DO NOT UPSERT
    # Remote paths / IDs
    namespace: str = "4de8f47d-4377-4f7b-9849-06dbd2e17e95"
    index_name: str = "test-financial-report-py"
    pinecone_environment: str = "asia-southeast1-gcp-free"

    # Local paths
    # TODO: double check this: data_root is relative to cwd.
    data_root: str = "examples/example_data/financial_report"
    metadata_db_path: str = "artifacts/metadata_db_py.json"
    portfolio_csv_path: str = "portfolios/client_a_portfolio.csv"

    # Misc
    client_name: str = "client_a"


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
