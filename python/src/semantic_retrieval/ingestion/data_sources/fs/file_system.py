from typing import Any, List, Optional, Callable

from result import Err, Ok, Result
from semantic_retrieval.ingestion.data_sources.data_source import DataSource
from semantic_retrieval.document.document import RawDocument, RawDocumentChunk


from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from semantic_retrieval.utils.callbacks import CallbackManager
import os
import hashlib
import uuid
import mimetypes


# TODO: (suyog) I dislike this quite a bit, but following typescript for now - same with the FileSystemRawDocument implementation of RawDocument
def csv_loader_func(path: str) -> CSVLoader:
    return CSVLoader(path)


def txt_loader_func(path: str) -> TextLoader:
    return TextLoader(path)


def pdf_loader_func(path: str) -> PyPDFLoader:
    return PyPDFLoader(path)


def docx_loader_func(path: str) -> Docx2txtLoader:
    return Docx2txtLoader(path)


DEFAULT_FILE_LOADERS: dict[str, Callable[[str], BaseLoader]] = {
    ".csv": csv_loader_func,
    ".txt": txt_loader_func,
    ".py": txt_loader_func,
    ".pdf": pdf_loader_func,
    ".docx": docx_loader_func,
}


class FileSystemRawDocument(RawDocument):
    file_loaders: dict[str, Callable[[str], BaseLoader]] = DEFAULT_FILE_LOADERS

    def __init__(
        self,
        file_loaders: Optional[dict[str, Callable[[str], BaseLoader]]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if file_loaders is not None:
            self.file_loaders = file_loaders

    async def get_content(self) -> Result[str, str]:
        # Get file loader w/ filePath (which is self.uri) & load_chunked_content
        _, file_extension = os.path.splitext(self.uri)

        if file_extension in self.file_loaders:
            file_loader = self.file_loaders[file_extension]
            loader = file_loader(self.uri)
            return Ok(loader.load()[0].page_content)
        else:
            return Err(f"File extension {file_extension} not supported")

    async def get_chunked_content(self) -> List[RawDocumentChunk]:
        # TODO: Implement later - not the same because lazy_load in langchain python is different
        return []


class FileSystem(DataSource):
    name: str = "FileSystem"
    path: str
    collection_id: Optional[str] = None
    callback_manager: Optional[CallbackManager] = None
    file_loaders: dict[str, Callable[[str], BaseLoader]] = DEFAULT_FILE_LOADERS

    def __init__(
        self,
        path: str,
        collection_id: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        file_loaders: Optional[dict[str, Callable[[str], BaseLoader]]] = None,
    ):
        self.path = path
        self.collection_id = collection_id
        self.callback_manager = callback_manager
        if file_loaders is not None:
            self.file_loaders = file_loaders

    def check_stats(self):
        return os.path.isdir(self.path), os.path.isfile(self.path)

    def load_file(self, path: str, collection_id: str) -> FileSystemRawDocument | None:
        file_name_with_ext = os.path.basename(path)
        file_name = os.path.splitext(file_name_with_ext)[0]
        # TODO: This should be done outside of python
        hash = hashlib.md5(open(path, "rb").read()).hexdigest()

        return FileSystemRawDocument(
            file_loaders=self.file_loaders,
            uri=path,
            data_source=self,
            name=file_name,
            mime_type=mimetypes.guess_type(path)[0],
            hash=hash,
            blob_id=None,
            document_id=str(uuid.uuid4()),
            collection_id=collection_id,
        )

    def load_documents(
        self, filters: Optional[Any] = None, limit: Optional[int] = None
    ) -> List[RawDocument]:
        # TODO: Filters & Limit are not implemented yet

        # Iterate through directory or just load a single file & make a list, handle error conditions like can't find file or directory
        isdir, isfile = self.check_stats()
        raw_documents = []

        if isdir:
            files = [f for f in os.listdir(self.path)]
            collection_id = (
                self.collection_id if self.collection_id else str(uuid.uuid4())
            )
            for file in files:
                subdir_path = os.path.join(self.path, file)
                if os.path.isdir(subdir_path):
                    subDir = FileSystem(subdir_path, collection_id)
                    subDir.load_documents()
                elif os.path.isfile(subdir_path):
                    raw_documents.append(self.load_file(subdir_path, collection_id))
        elif isfile:
            collection_id = (
                self.collection_id if self.collection_id else str(uuid.uuid4())
            )
            raw_documents.append(self.load_file(self.path, collection_id))
        else:
            message = f"{self.path} is neither a file nor a directory."
            err = Exception(message)

            # TODO: callback
            # event = LoadDocumentsErrorEvent(error=err)

            # # Call the method on the callback managerwith event as parameter
            # if self.callback_manager:
            #     asyncio.run(self.callback_manager.run_callbacks(event))

            raise Exception(err)

        # TODO: callback
        # event = LoadDocumentsSuccessEvent(raw_documents=raw_documents)

        # if self.callback_manager:
        #     asyncio.run(self.callback_manager.run_callbacks(event))

        return raw_documents
