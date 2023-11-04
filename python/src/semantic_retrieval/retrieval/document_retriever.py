from abc import abstractmethod
from typing import TypeVar, Optional, List

from result import Ok, Result
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.document.document import Document, DocumentFragment

from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.retrieval.retriever import (
    BaseRetriever,
    BaseRetrieverQueryParams,
)
from semantic_retrieval.utils.callbacks import CallbackManager

R = TypeVar("R")
Q = TypeVar("Q")


class DocumentRetriever(BaseRetriever[R, Q]):
    def __init__(
        self,
        metadata_db: DocumentMetadataDB,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(metadata_db, callback_manager)
        self.metadata_db = metadata_db

    @abstractmethod
    async def get_fragments_unsafe(self, params: BaseRetrieverQueryParams[Q]) -> List[DocumentFragment]:  # type: ignore [fixme]
        pass

    async def filter_accessible_fragments(
        self, fragments: List[DocumentFragment], access_passport: AccessPassport
    ) -> List[DocumentFragment]:
        accessible_fragments = []
        # TODO [P0] re-implement this
        return fragments
        for fragment in fragments:
            metadata = await self.metadata_db.get_metadata(fragment.document_id)

            if metadata and metadata.access_policies and metadata.document:  # type: ignore [fixme]
                policy_checks = []

                for policy in metadata.access_policies:  # type: ignore [fixme]
                    policy_checks.append(
                        await policy.test_document_read_permission(
                            metadata.unwrap().document,
                            access_passport.get_identity(policy.resource)
                            if policy.resource
                            else None,
                        )
                    )

                if False in policy_checks:
                    accessible_fragments.append(None)
            else:
                accessible_fragments.append(fragment)

        filtered_fragments = list(filter(None, accessible_fragments))

        return filtered_fragments

    async def get_documents_for_fragments(self, fragments: List[DocumentFragment]):  # type: ignore [fixme]
        # TODO [P0] impl
        documents = []

        return documents

    @abstractmethod
    async def process_documents(self, _documents: List[Document]) -> R:  # type: ignore [fixme]
        pass

    async def retrieve_data(
        self, params: BaseRetrieverQueryParams[Q]
    ) -> Result[R, str]:
        unsafe_fragments = await self.get_fragments_unsafe(params)
        accessible_fragments = await self.filter_accessible_fragments(
            unsafe_fragments, params.access_passport
        )
        accessible_documents = await self.get_documents_for_fragments(
            accessible_fragments
        )
        processed_documents = await self.process_documents(accessible_documents)

        return Ok(processed_documents)
