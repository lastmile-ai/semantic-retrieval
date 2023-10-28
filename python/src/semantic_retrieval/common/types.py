# Don't rely on the generic type. Wrong annotation might be missed.
# Use `Any` to signal that uncertainty explicitly.
from typing import Any

import numpy.typing as npt
from pydantic import BaseModel

# TODO: is this useful?
NPA = npt.NDArray[Any]

ArrayLike = npt.ArrayLike


class Record(BaseModel):
    class Config:
        strict = True
        frozen = True


class A:
    x = 1


class B:
    y = 2


C = A | B


def processC(c: C):
    match c:
        case A(x=_):
            print("the A=", c.x)
        case B(y=y_):
            print(y_)


class User(Record):
    id: int
    name: str = "Jane Doe"


class Cat(Record):
    meows: int


class Dog(Record):
    barks: float


class Lizard(Record):
    scales: bool


Animal = Cat | Dog | Lizard


class Model(Record):
    pet: Animal
    n: int


# EXAMPLES
# from semantic_retrieval.common.types import Record


# def test_py_union_matching():
#     print("test_py_union_matching")
#     processC(A())
#     processC(B())


# def processAnimal(a: Animal):
#     print("processAnimal")
#     match a:
#         case Cat(meows=m_):
#             print(m_)
#         case Dog(barks=b_):
#             print(b_)
#         case Lizard(scales=s_):
#             print(s_)


# def test_pydantic_union_matching():
#     print("test_pydantic_union_matching")
#     c = Cat(meows=1)
#     m = Model(pet=c, n=4)
#     print(m)
#     processAnimal(c)


# def test_pydantic_record():
#     print("test_pydantic_record")
#     user = User(id=123)

#     print(user)


# def main():
#     test_py_union_matching()
#     test_pydantic_union_matching()
#     test_pydantic_record()
