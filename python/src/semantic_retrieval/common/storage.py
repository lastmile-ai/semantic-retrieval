from abc import ABC, abstractmethod
from typing import Optional


class BlobStorage(ABC):
    @abstractmethod
    def write(self, blob: bytes, name: Optional[str]) -> "BlobIdentifier":
        pass

    @abstractmethod
    def write_stream(self, stream, name: Optional[str]) -> "BlobIdentifier":
        pass

    @abstractmethod
    def read(self, blob_uri: str) -> bytes:
        pass

    @abstractmethod
    def read_stream(self, blob_uri: str):
        pass

    @abstractmethod
    def delete(self, blob_uri: str):
        pass


class BlobIdentifier(ABC):
    blob_uri: str
    storage: BlobStorage
