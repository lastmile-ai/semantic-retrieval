from typing import Union, List, Dict

# Define a JSONValue as a union of common JSON value types
JSONValue = Union[
    str,
    int,
    bool,
    Dict[str, Union[None, str, int, bool, Dict[str, "JSONValue"]]],
    List["JSONValue"],
]

# Define a JSONObject as a dictionary with string keys and JSONValue values
JSONObject = Dict[str, Union[None, str, int, bool, Dict[str, JSONValue]]]

# Define a JSONArray as a list of JSONValue elements
JSONArray = List[JSONValue]
