# from typing import Any, Dict, TypeVar
from semantic_retrieval.common.types import Record


class Config(Record):
    namespace: str = "ns123"
    index_name: str = "test-financial-report-py"
    data_root: str = "examples/example_data/financial_report"
    metadata_db_path: str = "artifacts/metadata_db_py.json"
