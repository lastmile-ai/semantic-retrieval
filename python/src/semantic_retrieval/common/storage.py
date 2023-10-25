
from abc import ABC, abstractmethod


class BlobStorage(ABC):
    @abstractmethod
    def write(self, blob: bytes, name: str | None) -> 'BlobIdentifier':
        pass

    @abstractmethod
    def write_stream(self, stream, name: str | None) -> 'BlobIdentifier':
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

