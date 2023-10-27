from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
    RetrieverFilterAccessibleFragmentsEvent,
    RetrieverGetDocumentsForFragmentsEvent,
    RetrieveDataEvent,
)

from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB


class BaseRetriever(Traceable):
    metadata_db: DocumentMetadataDB
    callback_manager: CallbackManager

    def __init__(self, metadata_db: DocumentMetadataDB):
        self.metadata_db = metadata_db

    async def get_fragments_unsafe(self, params):
        return []

    async def filter_accessible_fragments(self, fragments, access_passport=None):
        if not self.metadata_db:
            return fragments

        accessible_fragments = []
        for fragment in fragments:
            metadata = self.metadata_db.get_metadata(fragment.documentId)

            if not metadata:
                accessible_fragments.append(fragment)
                continue

            if metadata.accessPolicies and metadata.document:
                policy_checks = []
                for policy in metadata.accessPolicies:
                    identity = None
                    if policy.resource:
                        identity = access_passport.get_identity(policy.resource)

                    policy_checks.append(
                        policy.test_document_read_permission(
                            metadata.document, identity
                        )
                    )

                if any(check is False for check in policy_checks):
                    continue

            accessible_fragments.append(fragment)

        event = RetrieverFilterAccessibleFragmentsEvent(
            name="onRetrieverFilterAccessibleFragments", fragments=accessible_fragments
        )
        await self.callback_manager.run_callbacks(event)

        return accessible_fragments

    async def get_documents_for_fragments(self, fragments):
        fragments_by_document_id = {}
        for fragment in fragments:
            if not fragment:
                continue
            if fragment.documentId not in fragments_by_document_id:
                fragments_by_document_id[fragment.documentId] = []
            fragments_by_document_id[fragment.documentId].append(fragment)

        documents = []
        for document_id, fragments in fragments_by_document_id.items():
            document_metadata = self.metadata_db.get_metadata(document_id)
            stored_document = document_metadata["document"]
            del document_metadata["document"]

            def serialize_fragments():
                serialized_fragments = []
                for fragment in fragments:
                    serialized_fragments.append(fragment.serialize())
                return "\n".join(serialized_fragments)

            def serialize_document():
                if stored_document:
                    return stored_document.serialize()
                else:
                    serialized_fragments = serialize_fragments()
                    file_path = f"{document_id}.txt"
                    with open(file_path, "w") as file:
                        file.write(serialized_fragments)
                    return file_path

            document = {
                **document_metadata,
                "documentId": document_id,
                "fragments": fragments,
                "attributes": document_metadata.get("attributes", {}),
                "metadata": document_metadata.get("metadata", {}),
                "serialize": serialize_document,
            }

            documents.append(document)

        event = RetrieverGetDocumentsForFragmentsEvent(
            name="onRetrieverGetDocumentsForFragments", documents=documents
        )
        await self.callback_manager.run_callbacks(event)

        return documents

    def process_documents(self, documents):
        raise NotImplementedError

    async def retrieve_data(self, params):
        unsafe_fragments = await self.get_fragments_unsafe(params)

        accessible_fragments = await self.filter_accessible_fragments(
            unsafe_fragments, params["accessPassport"]
        )

        accessible_documents = await self.get_documents_for_fragments(
            accessible_fragments
        )

        processed_documents = await self.process_documents(accessible_documents)

        event = RetrieveDataEvent(name="onRetrieveData", data=processed_documents)
        await self.callback_manager.run_callbacks(event)

        return processed_documents
