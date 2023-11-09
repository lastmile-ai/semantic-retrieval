import argparse
from typing import Any, Dict, List, Optional, Set, Type

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


def add_parser_argument(parser, field_name, field, is_required: bool):  # type: ignore
    field_name = field_name.replace("_", "-")
    the_type = field.annotation
    parser.add_argument(f"--{field_name}", type=the_type, required=is_required)


def add_parser_arguments(parser, fields, required: Optional[Set[str]] = None):  # type: ignore
    required = required or set()
    for field_name, field in fields.items():
        is_required = field_name in required
        add_parser_argument(parser, field_name, field, is_required)


def argparsify(r: Record | Type[Record], required: Optional[Set[str]] = None):
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser, r.model_fields, required=required)
    return parser
