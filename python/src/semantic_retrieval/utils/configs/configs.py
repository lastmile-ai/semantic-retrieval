import argparse
from typing import Any, Dict, List, Type

from semantic_retrieval.common.types import Record


def remove_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def combine_dicts(d_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Precedence: highest first
    """
    out = {}
    for d in reversed(d_list):
        out.update(d)

    return out


def add_parser_argument(parser, field_name, field):  # type: ignore
    field_name = field_name.replace("_", "-")
    the_type = field.annotation
    parser.add_argument(f"--{field_name}", type=the_type)


def add_parser_arguments(parser, fields):  # type: ignore
    for field_name, field in fields.items():
        add_parser_argument(parser, field_name, field)


def argparsify(r: Record | Type[Record]):
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser, r.model_fields)
    return parser
