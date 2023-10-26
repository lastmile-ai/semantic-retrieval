# Don't rely on the generic type. Wrong annotation might be missed.
# Use `Any` to signal that uncertainty explicitly.
from typing import Any
import numpy.typing as npt


# TODO: is this useful?
NPA = npt.NDArray[Any]

ArrayLike = npt.ArrayLike
