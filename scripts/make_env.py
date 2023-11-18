import argparse
from collections import OrderedDict
import re
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import sys

from result import Err, Ok, Result

from semantic_retrieval.functional.functional import result_reduce_list_all_ok


def read_csv(csv_file: str) -> Result[OrderedDict[str, str], str]:
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return Err(f"Invalid CSV file: {csv_file}")

    df_filtered = df[df.note.str.contains(".env").fillna(False)]

    def _normalize(s: str) -> Result[str, str]:
        m = re.match(r"(.*)\.(.*)", s)
        if not m:
            return Err(f"Invalid name: {s}")

        return Ok(m.groups()[0].upper())

    s_keys_normed = df_filtered["name"].apply(_normalize).tolist()
    res_keys = result_reduce_list_all_ok(s_keys_normed)
    values = df_filtered["password"]

    match res_keys:
        case Err(e):
            return Err(e)
        case Ok(keys):
            return Ok(OrderedDict(sorted(zip(keys, values))))


def read_env(env_file: str) -> Result[OrderedDict[str, str], str]:
    with open(env_file) as f:
        result = {}
        for line in f:
            line = line.strip()
            try:
                k, v = line.split("=", 1)
                result[k] = v
            except ValueError:
                return Err(f"Invalid line: {line}")
        return Ok(OrderedDict(sorted(result)))


def merge_OrderedDicts(
    csv_OrderedDict: OrderedDict[str, str], env_OrderedDict: OrderedDict[str, str]
) -> OrderedDict[str, str]:
    for key, value in csv_OrderedDict.items():
        if key in env_OrderedDict and value != env_OrderedDict[key]:
            print(f"Key: {key} has different values in .env and csv files")
            sys.exit()

    env_OrderedDict.update(csv_OrderedDict)
    return env_OrderedDict


def write_env(output_path: str, env_OrderedDict: OrderedDict[str, str]) -> int:
    with open(output_path, "w") as f:
        for key, value in env_OrderedDict.items():
            f.write(f"{key}={value}\n")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvpath", required=True)
    parser.add_argument("--envpath")
    args = parser.parse_args()

    res_csv_OrderedDict = read_csv(args.csvpath)

    if args.envpath:
        result = _run_with_env_path(args.envpath, res_csv_OrderedDict)
    else:
        result = _run_with_csv_only(res_csv_OrderedDict)

    print(f"Done: (return value: {result})")


def _run_with_csv_only(res_csv_OrderedDict: Result[OrderedDict[str, str], str]):
    # create new .env file
    match res_csv_OrderedDict:
        case Err(e):
            print(e)
            return 1
        case Ok(csv_OrderedDict):
            return write_env(".env", csv_OrderedDict)


def _run_with_env_path(
    envpath: str, res_csv_OrderedDict: Result[OrderedDict[str, str], str]
):
    # overload existing .env file
    load_dotenv(find_dotenv(envpath))

    res_env_OrderedDict = read_env(envpath)

    match res_csv_OrderedDict:
        case Err(e):
            print(e)
            return 1
        case Ok(csv_OrderedDict):
            return write_env_for_env_OrderedDict(csv_OrderedDict, res_env_OrderedDict)


def write_env_for_env_OrderedDict(
    csv_OrderedDict: OrderedDict[str, str],
    res_env_OrderedDict: Result[OrderedDict[str, str], str],
):
    match res_env_OrderedDict:
        case Err(e):
            print(e)
            return 1
        case Ok(env_OrderedDict):
            new_env_OrderedDict = merge_OrderedDicts(csv_OrderedDict, env_OrderedDict)
            return write_env(".env", new_env_OrderedDict)


if __name__ == "__main__":
    res = main()
    sys.exit(res)
