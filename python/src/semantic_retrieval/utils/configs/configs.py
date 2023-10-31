from typing import Any, Dict, List


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
