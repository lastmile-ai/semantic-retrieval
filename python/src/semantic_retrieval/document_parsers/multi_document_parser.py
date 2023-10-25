from semantic_retrieval.document.document import RawDocument, IngestedDocument


class ParserConfig:
    metadata_db: any | None  # TODO: Add type for metadata db in followup diff
    access_control_policy_factory: any | None  # TODO: Add type for access control policy factory in followup diff
    parser_registry: any | None  # TODO: Add type for parser registry in followup diff

    def __init__(
        self, metadata_db=None, access_control_policy_factory=None, parser_registry=None
    ):
        self.metadata_db = metadata_db
        self.access_control_policy_factory = access_control_policy_factory
        self.parser_registry = parser_registry


async def parse_documents(
    self, documents: list[RawDocument], parser_config: ParserConfig
) -> list[IngestedDocument]:
    return documents
