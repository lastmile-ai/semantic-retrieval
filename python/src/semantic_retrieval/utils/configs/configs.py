import argparse
import logging
import os
from enum import EnumMeta
from types import UnionType
from typing import Any, Dict, Optional, Set, Type, TypeVar

import pydantic
from dotenv import load_dotenv
from result import Ok, Result
from semantic_retrieval.common.core import (
    dict_union_allow_replace,
    remove_nones,
)
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import Record
from semantic_retrieval.functional.functional import ErrWithTraceback

LOGGER = logging.getLogger(__name__)


def add_parser_argument(
    parser: argparse.ArgumentParser,
    field_name: str,
    field: pydantic.fields.FieldInfo,
    is_required: bool,
) -> Optional[str]:
    field_name = field_name.replace("_", "-")
    the_type = field.annotation
    if the_type is None:
        return f"{field_name}: the_type is None"
    elif the_type is bool:
        if is_required:
            return f"(bool) flag cannot be required: field={field_name}"
        parser.add_argument(
            f"--{field_name}",
            action="store_true",
        )
    elif type(the_type) is UnionType:
        return f"UnionType not supported. field={field_name}"
    elif isinstance(the_type, EnumMeta):
        parser.add_argument(
            f"--{field_name}",
            type=str,
            choices=[e.value.lower() for e in the_type],  # type: ignore
            required=is_required,
        )
    else:
        parser.add_argument(
            f"--{field_name}", type=the_type, required=is_required
        )


def add_parser_arguments(
    parser: argparse.ArgumentParser,
    fields: dict[str, pydantic.fields.FieldInfo],
    required: Optional[Set[str]] = None,
):
    required = required or set()
    for field_name, field in fields.items():
        is_required = field_name in required
        res = add_parser_argument(parser, field_name, field, is_required)
        if res is not None:
            LOGGER.warning(res)


def argparsify(
    r: Record | Type[Record], required: Optional[Set[str]] = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser, r.model_fields, required=required)
    return parser


def get_config_args(
    args: argparse.Namespace | Dict[str, Any]
) -> Dict[str, Any]:
    def _get_cli():
        if isinstance(args, argparse.Namespace):
            return vars(args)
        else:
            return dict(args)

    env_keys = {
        "openai_key": "OPENAI_API_KEY",
        "pinecone_key": "PINECONE_API_KEY",
        "pinecone_environment": "PINECONE_ENVIRONMENT",
        "pinecone_index_name": "PINECONE_INDEX_NAME",
        "pinecone_namespace": "PINECONE_NAMESPACE",
    }

    load_dotenv()
    env_values = remove_nones({k: os.getenv(v) for k, v in env_keys.items()})
    cli_values = remove_nones(_get_cli())

    all_args = dict_union_allow_replace(
        env_values, cli_values, on_conflict="replace"
    )
    return all_args


def resolve_path(data_root: str, path: str) -> str:
    """
    data_root: path relative to cwd where all the data is stored.
    path: relative to data_root, the specific data to look at,
    e.g. "10ks/".
    """

    joined = os.path.join(os.getcwd(), data_root, path)
    return os.path.abspath(os.path.expanduser(joined))


T_Record = TypeVar("T_Record", bound=Record)


def parse_args(
    parser: argparse.ArgumentParser,
    argv: list[str],
    config_type: Type[T_Record],
) -> Result[T_Record, str]:
    cli_dict = remove_nones(vars(parser.parse_args(argv)))
    return config_from_primitives(config_type, cli_dict)


def config_from_primitives(
    config_type: Type[T_Record], cli_dict: JSONObject
) -> Result[T_Record, str]:
    try:
        return Ok(config_type.model_validate(cli_dict))
    except pydantic.ValidationError as e:
        return ErrWithTraceback(e, "Invalid args")
