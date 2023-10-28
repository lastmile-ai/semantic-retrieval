from typing import Any, Optional, Callable
from semantic_retrieval.ingestion.data_sources.data_source import DataSource
from semantic_retrieval.document.document import RawDocument, RawDocumentChunk


from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader

from semantic_retrieval.utils.callbacks import CallbackManager
import os

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
    ".pdf": pdf_loader_func,
    ".docx": docx_loader_func,
}

class FileSystemRawDocument(RawDocument):
    file_loaders: dict[str, Callable[[str], BaseLoader]] = DEFAULT_FILE_LOADERS

    def __init__(self, file_loaders: Optional[dict[str, Callable[[str], BaseLoader]]] = None, **kwargs: Any):
        super().__init__(**kwargs) 
        if file_loaders is not None:
            self.file_loaders = file_loaders

    def get_content(self) -> str | None:
        # Get file loader w/ filePath (which is self.uri) & load_chunked_content
        _, file_extension = os.path.splitext(self.url)
        
        if file_extension in self.file_loaders:
            file_loader = self.file_loaders[file_extension]
            loader = file_loader(self.url)
            return loader.load()[0].page_content

    def get_chunked_content(self) -> list[RawDocumentChunk]:
        # TODO: Implement later - not the same because lazy_load in langchain python is different
        return []


class FileSystem(DataSource):
    name: str = "FileSystem"
    path: str
    collection_id: Optional[str] = None
    callback_manager: Optional[CallbackManager] = None
    file_loaders: dict[str, Callable[[str], BaseLoader]] = DEFAULT_FILE_LOADERS

    def __init__(self, path: str, collection_id: Optional[str] = None, callback_manager: Optional[CallbackManager] = None, file_loaders: Optional[dict[str, Callable[[str], BaseLoader]]] = None):
        self.path = path
        self.collection_id = collection_id
        self.callback_manager = callback_manager
        if file_loaders is not None:
            self.file_loaders = file_loaders

    def load_file(self) -> FileSystemRawDocument | None:
        return FileSystemRawDocument(
            file_loaders=self.file_loaders,
            url=self.path,
            data_source=self,
            name=self.path,
            mime_type="",
            hash="",
            blob_id=None,
            document_id="",
            collection_id=self.collection_id,
        )

    def load_documents(
        self, filters: Optional[Any] = None, limit: Optional[int] = None
    ) -> list[RawDocument]:
        # TODO: Filters & Limit are not implemented yet

        # Iterate through directory or just load a single file & make a list, handle error conditions like can't find file or directory

        return []
