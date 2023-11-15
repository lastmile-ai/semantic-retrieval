import argparse
import os
from typing import Any, Dict, List, Optional, Set, Type

from dotenv import load_dotenv

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


def get_config_args(args: argparse.Namespace | Dict[str, Any]) -> Dict[str, Any]:
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

    # TODO: generalize combining args
    all_args = dict(env_values, **cli_values)
    return all_args


def resolve_path(data_root: str, path: str) -> str:
    """
    data_root: path relative to cwd where all the data is stored.
    path: relative to data_root, the specific data to look at,
    e.g. "10ks/".
    """

    joined = os.path.join(os.getcwd(), data_root, path)
    return os.path.abspath(os.path.expanduser(joined))
