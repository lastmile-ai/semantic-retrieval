import glob
import logging
import os
import shlex
import subprocess
import sys
from enum import Enum
from typing import Any, List, Optional, Sequence, Union

from pydantic import field_validator
from result import Err, Ok, Result
from semantic_retrieval.common.core import (
    combine_returncodes,
    load_json_file,
    normalize_path,
)
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import Record
from semantic_retrieval.functional.functional import (
    result_reduce_list_all_ok,
    result_to_exitcode,
)
from semantic_retrieval.utils.configs.configs import argparsify, parse_args

LOGGER = logging.getLogger(__name__)


class Mode(Enum):
    FIX = "FIX"
    LIST_FILES = "LIST_FILES"
    CHECK_FAST = "CHECK_FAST"
    CHECK = "CHECK"


class BlackCheck(Record):
    line_length: int
    diff_only: bool


class PyrightCheck(Record):
    pass


class PylintCheck(Record):
    args: List[str]


class ISortCheck(Record):
    line_length: int


Check = BlackCheck | PyrightCheck | PylintCheck | ISortCheck


class Config(Record):
    mode: Mode = Mode.CHECK_FAST
    verbose: bool = False
    files: Optional[List[str]] = None
    vscode_settings_path: str = ".vscode/settings.json"
    path_glob_excludes: str = ".lint_glob_excludes"

    @field_validator("mode", mode="before")
    def convert_to_mode(  # pylint: disable=no-self-argument
        cls, value: Any
    ) -> Mode:
        if isinstance(value, str):
            try:
                return Mode[value.upper()]
            except KeyError as e:
                raise ValueError(f"Unexpected value for mode: {value}") from e
        return value


def read_glob_excludes(config_file: str) -> List[str]:
    if (
        not config_file
        or not os.path.isfile(config_file)
        or not os.access(config_file, os.R_OK)
    ):
        LOGGER.warning(
            "Could not read glob excludes file: '%s'. "
            + "Check that the file exists and that it has the correct format.",
            config_file,
        )
        return []
    with open(config_file, "r", encoding="utf8") as f:
        lines = f.readlines()
        out = [line.strip() for line in lines if line.strip()]
        return out


def get_python_files() -> List[str]:
    return glob.glob("./**/*.py", recursive=True)


def get_files_exclude(glob_excludes: List[str]) -> List[str]:
    out = []
    for g in glob_excludes:
        paths = glob.glob(g, recursive=True)
        out += paths

    return out


def get_files_without_excludes(
    files: List[str], files_exclude: List[str]
) -> List[str]:
    files_exclude_norm = [os.path.realpath(f) for f in files_exclude]
    return [f for f in files if os.path.realpath(f) not in files_exclude_norm]


def run_lint_cmd(cmd: Union[str, List[str]], files: List[str]) -> int:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    run_output = subprocess.run(
        cmd + files,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    sys.stderr.write(run_output.stderr or "")
    sys.stdout.write(run_output.stdout or "")
    return run_output.returncode


def main(argv: List[str]) -> int:
    parser = argparsify(Config)
    res_cfg = parse_args(parser, argv[1:], Config)

    match res_cfg:
        case Err(e):
            LOGGER.critical("err: %s", e)
            return result_to_exitcode(Err(e))
        case Ok(cfg):
            python_files_to_lint = _get_files_to_lint(
                cfg.files, cfg.path_glob_excludes
            )

            if not python_files_to_lint:
                print(
                    """No files to lint.
                    * If --files was given, check the paths.
                    * Also check lint glob excludes file."""
                )
                return 1

            final_result = _get_final_result(
                cfg.mode,
                python_files_to_lint,
                cfg.vscode_settings_path,
                cfg.verbose,
            )
            return result_to_exitcode(final_result)


def _get_final_result(
    mode: Mode,
    python_files_to_lint: list[str],
    vscode_settings_path: str | None,
    verbose: bool,
) -> Result[int, str]:
    settings = load_json_file(vscode_settings_path)
    line_length = settings.and_then(_extract_line_length)
    pylint_args = settings.and_then(_get_pylint_args)

    match mode:
        case Mode.LIST_FILES:
            print("\n".join(python_files_to_lint))
            return Ok(0)
        case Mode.FIX:
            return line_length.and_then(
                lambda ll: run_with(  # type: ignore
                    [
                        ISortCheck(line_length=ll),
                        BlackCheck(line_length=ll, diff_only=False),
                    ],
                    python_files_to_lint,
                    verbose,
                )
            )
        case Mode.CHECK_FAST:
            return line_length.and_then(
                lambda ll: run_with(  # type: ignore
                    [
                        BlackCheck(line_length=ll, diff_only=True),
                        PyrightCheck(),
                    ],
                    python_files_to_lint,
                    verbose,
                )
            )
        case Mode.CHECK:
            return pylint_args.and_then(
                lambda pyla: line_length.and_then(  # type: ignore
                    lambda ll: run_with(  # type: ignore
                        [
                            BlackCheck(line_length=ll, diff_only=True),
                            PyrightCheck(),
                            PylintCheck(args=pyla),
                        ],
                        python_files_to_lint,
                        verbose,
                    )
                )
            )


def run_with(
    check: Sequence[Check], files: list[str], verbose: bool
) -> Result[int, str]:
    def _run_with_one(check: Check) -> Result[int, str]:
        match check:
            case BlackCheck(line_length=line_length, diff_only=diff_only):
                return run_with_black(files, line_length, diff_only, verbose)
            case PyrightCheck():
                return run_with_pyright(files)
            case PylintCheck(args=args):
                return run_with_pylint(files, args, verbose)
            case ISortCheck(line_length=line_length):
                return run_with_isort(files, line_length)

    check_result = result_reduce_list_all_ok(map(_run_with_one, check))
    return check_result.map(combine_returncodes)


def run_with_black(
    files: list[str], line_length: int, diff_only: bool, verbose: bool
) -> Result[int, str]:
    print("Running black")

    cmd = ["black"]
    if diff_only:
        cmd.append("--diff")
    if line_length:
        cmd.extend(["--line-length", str(line_length)])
    if verbose:
        cmd.append("-v")

    code = run_lint_cmd(cmd, files)
    return Ok(code)


def run_with_pyright(files: list[str]) -> Result[int, str]:
    print("Running pyright")
    cmd = ["pyright"]
    code = run_lint_cmd(cmd, files)
    return Ok(code)


def run_with_pylint(
    files: list[str], pylint_args: List[str], verbose: bool
) -> Result[int, str]:
    print("Running pylint")
    cmd = ["pylint"] + pylint_args
    if verbose:
        cmd.append("-v")
    code = run_lint_cmd(cmd, files)
    return Ok(code)


def run_with_isort(files: list[str], line_length: int) -> Result[int, str]:
    print("Running isort")
    cmd = [
        "isort",
        "--profile=black",
        "--line-length",
        str(line_length),
    ]
    code = run_lint_cmd(cmd, files)
    return Ok(code)


def _get_files_to_lint(
    files: list[str] | None, path_glob_excludes: str
) -> List[str]:
    files = files if files is not None else get_python_files()
    files_before_excludes = list(map(normalize_path, files))
    glob_excludes = read_glob_excludes(path_glob_excludes)
    files_exclude = list(map(normalize_path, get_files_exclude(glob_excludes)))

    python_files_to_lint = list(
        set(files_before_excludes) - set(files_exclude)
    )

    return python_files_to_lint


def lint_only(
    pylint_args: List[str],
    python_files_to_lint: List[str],
    verbose: bool,
    line_length: int,
) -> Result[int, str]:
    print("Running pylint")
    pylint_cmd = ["pylint"] + pylint_args
    if verbose:
        pylint_cmd.append("-v")
    pylint_code = run_lint_cmd(pylint_cmd, python_files_to_lint)

    print("Running black (show diff)")

    black_cmd = ["black", "--diff"]
    if line_length:
        black_cmd.extend(["--line-length", str(line_length)])
    if verbose:
        black_cmd.append("-v")

    black_code = run_lint_cmd(black_cmd, python_files_to_lint)
    returncode = combine_returncodes([pylint_code, black_code])
    return Ok(returncode)


def _extract_line_length(settings: JSONObject) -> Result[int, str]:
    black_args: List[str] = settings.get(
        "black-formatter.args", []
    )  # type: ignore
    for arg in black_args:
        if arg.startswith("--line-length"):
            return Ok(int(arg.split("=")[1]))
    return Err("Could not find line length in settings")


def _get_pylint_args(settings: JSONObject) -> Result[List[str], str]:
    if "pylint.args" in settings:
        return Ok(settings["pylint.args"])  # type: ignore
    else:
        return Err("Could not find pylint args in settings")


if __name__ == "__main__":
    retcode: int = main(sys.argv)
    sys.exit(retcode)
