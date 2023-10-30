from dataclasses import dataclass
from semantic_retrieval.document.document import RawDocument, IngestedDocument
from typing import Any, Optional, Sequence

from semantic_retrieval.ingestion.document_parsers.document_parser import DocumentParser


@dataclass
class ParserConfig:
    metadata_db: Optional[Any]  # TODO: Add type for metadata db in followup diff
    access_control_policy_factory: Optional[
        Any
    ]  # TODO: Add type for access control policy factory in followup diff
    parser_registry: Optional[Any] = None  # TODO: Add type for parser registry in followup diff


class MultiDocumentParser(DocumentParser):
    async def parse_documents(
        self, documents: Sequence[RawDocument], parser_config: ParserConfig
    ) -> Sequence[IngestedDocument]:  # type: ignore [fixme]
        # TODO impl
        pass
