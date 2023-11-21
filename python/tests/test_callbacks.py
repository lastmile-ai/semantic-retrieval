from dataclasses import dataclass

from semantic_retrieval.common.types import Record
from semantic_retrieval.utils.callbacks import (
    safe_serialize_arbitrary_for_logging,
)


@dataclass
class Person:
    name: str
    age: int
    secret_key: str


class Book(Record):
    title: str
    author: Person
    secret_key: str


def test_serialize_keys_redacted():
    # Test case 1
    person = Person("John", 23, "key1")
    inp = {"openai aPi_kEY": "234", "person": person}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    assert "234" not in out and "key1" not in out

    # Test case 2
    book = Book(title="Book Title", author=person, secret_key="key2")
    inp = {"openai aPi_kEY": "345", "book": book}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    assert "345" not in out and "key2" not in out

    # Test case 3
    inp = {"openai aPi_kEY": "456", "list": [person, book]}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    assert "456" not in out and "key1" not in out and "key2" not in out

    # Test case 4
    person_list = [
        Person("Person{}".format(i), i, "key{}".format(i)) for i in range(10)
    ]
    inp = {"openai aPi_kEY": "567", "list": person_list}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    for i in range(10):
        assert "567" not in out and "key{}".format(i) not in out

    # Test case 5
    book_list = [
        Book(
            title="Book{}".format(i),
            author=person_list[i],
            secret_key="key{}".format(i),
        )
        for i in range(10)
    ]
    inp = {"openai aPi_kEY": "678", "list": book_list}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    for i in range(10):
        assert "678" not in out and "key{}".format(i) not in out

    inp = {"openai aPi_kEY": "123"}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    assert "123" not in out

    inp = {"other_key": "789", "openai_aPi_kEY": "890"}
    out = safe_serialize_arbitrary_for_logging(inp)
    print(out)
    assert "890" not in out and "789" in out
