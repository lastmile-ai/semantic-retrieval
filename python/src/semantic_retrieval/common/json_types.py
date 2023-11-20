# Define a JSONValue as a union of common JSON value types

JSONPrimitive = str | int | bool | float | None

JSONDict = dict[str, "JSONValue"]
JSONList = list["JSONValue"]

JSONValue = JSONPrimitive | JSONList | JSONDict

JSONObject = JSONDict
