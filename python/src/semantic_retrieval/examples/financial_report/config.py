# from typing import Any, Dict, TypeVar
from semantic_retrieval.common.types import Record


class Config(Record):
    # DO NOT UPSERT
    namespace: str = "4de8f47d-4377-4f7b-9849-06dbd2e17e95"
    index_name: str = "test-financial-report-py"
    data_root: str = "examples/example_data/financial_report"
    metadata_db_path: str = "artifacts/metadata_db_py.json"
    pinecone_environment: str = "asia-southeast1-gcp-free"
    pinecone_key_path_abs: str = "/Users/jonathan/keys/dev_pinecone_key.txt"
    openai_key_path_abs: str = "/Users/jonathan/keys/dev_OPENAI_API_KEY.txt"
