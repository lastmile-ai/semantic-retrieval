import re


def to_camel_case(snake_str: str) -> str:
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_lower_camel_case(snake_str: str) -> str:
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return snake_str[0].lower() + camel_string[1:]


def canonical_field(s: str) -> str:
    # mapping = {
    #     "document_id": "documentId",
    #     "fragment_id": "fragmentId",
    # }
    # if s in mapping:
    #     return mapping[s]

    try:
        return to_lower_camel_case(s)
    except Exception:
        return s


def to_snake_case(s: str) -> str:
    try:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    except Exception:
        return s


def from_canonical_field(s: str) -> str:
    try:
        return to_snake_case(s)
    except Exception:
        return s
