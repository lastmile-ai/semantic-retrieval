from typing import Any, Optional, Callable
from semantic_retrieval.ingestion.data_sources.data_source import DataSource
from semantic_retrieval.document.document import RawDocument


from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader

from semantic_retrieval.utils.callbacks import CallbackManager

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

    def load_file(self):
        pass

    def load_documents(
        self, filters: Optional[Any] = None, limit: Optional[int] = None
    ) -> list[RawDocument]:
        return []
