import argparse
import re
import sys

import pandas as pd
from result import Err, Ok, Result
from result import do as result_do
from semantic_retrieval.common.core import (
    dict_union,
    file_contents,
    text_file_write,
)
from semantic_retrieval.functional.functional import (
    ErrWithTraceback,
    print_result,
    result_reduce_list_all_ok,
    result_to_exitcode,
)


def read_env_csv(csv_file: str) -> Result[dict[str, str], str]:
    """
    CSV expected to have very specific format, like
        name,password,note
        mykey.com,abc,.env
        somesite.com,123,my arbitrary note
    """
    try:
        df = pd.read_csv(csv_file)
        df = df[df.note.str.contains(".env").fillna(False)]
        names = df["name"].tolist()
        passwords = df["password"].tolist()
    except Exception as e:
        return ErrWithTraceback(
            e,
            extra_msg=f"Invalid CSV file: {csv_file}. See read_env_csv() docs.",
        )

    def _normalize_key(s: str) -> Result[str, str]:
        # remove .com etc. and uppercase
        m = re.match(r"(.*)\.(.*)", s)
        if not m:
            return Err(f"Invalid name: {s}")

        return Ok(m.groups()[0].upper())

    s_keys_normed = map(_normalize_key, names)
    res_keys = result_reduce_list_all_ok(s_keys_normed)
    return result_do(
        # create mapping
        Ok(dict(zip(keys, passwords)))
        # get the keys if they are ok
        for keys in res_keys
    )


def read_env(env_file: str) -> Result[dict[str, str], str]:
    def _process_contents(contents: str) -> Result[dict[str, str], str]:
        # parse .env file contents into a dict
        lines = [s.strip() for s in contents.split("\n") if s.strip()]
        lines_processed = result_reduce_list_all_ok(map(_process_line, lines))

        return lines_processed.map(dict)

    def _process_line(line: str) -> Result[tuple[str, str], str]:
        # parse a single line of .env file: my_key=123
        try:
            k, v = line.strip().split("=", 1)
            return Ok((k.strip(), v.strip()))
        except ValueError as e:
            return ErrWithTraceback(e, extra_msg=f"Invalid line: {line}")

    contents = file_contents(env_file)
    out = contents.and_then(_process_contents)
    return out


def fmt_orderdict(d: dict[str, str]) -> str:
    return "".join(
        # env file format (key=value)
        f"{key}={value}\n"
        # Sort for reproducibility
        for key, value in sorted(list(d.items()))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath", required=True)
    parser.add_argument("--envpath")
    args = parser.parse_args()

    res_csv_dict = read_env_csv(args.csvpath)

    final_result: Result[str, str] = (
        _run_with_env_path(args.envpath, res_csv_dict)
        if args.envpath
        else res_csv_dict.and_then(write)
    )

    print("Final result:")
    print_result(final_result)
    return result_to_exitcode(final_result)


def write(env_dict: dict[str, str], path: str = ".env") -> Result[str, str]:
    # helper to format dict, write to .env
    written = text_file_write(path, fmt_orderdict(env_dict))
    return result_do(
        Ok(f"Success! Wrote {n} bytes.")
        # n is the number of bytes written
        for n in written
    )


def _run_with_env_path(
    envpath: str, res_csv_dict: Result[dict[str, str], str]
) -> Result[str, str]:
    res_env_dict = read_env(envpath)

    return result_do(
        write(new_env_dict)
        for csv_dict in res_csv_dict
        for env_dict in res_env_dict
        for new_env_dict in dict_union(csv_dict, env_dict, on_conflict="err")
    )


if __name__ == "__main__":
    res = main()
    sys.exit(res)
