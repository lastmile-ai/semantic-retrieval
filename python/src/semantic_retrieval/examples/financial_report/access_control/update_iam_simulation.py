"""
This script simulates updating IAM permissions
which are used to control access to portfolios and 10ks.

For the simulation, we store the permissions in a local JSON file.
"""

import argparse
import json
import sys

from semantic_retrieval.common.core import file_contents, text_file_write


def main():
    parser = argparse.ArgumentParser()
    # create a subparser
    subparsers = parser.add_subparsers(required=True)
    # create a subparser for the "update" command
    parser_advisor_roles = subparsers.add_parser(
        "update_advisor_iam_role",
        help="""
        Update (simulated) IAM advisor roles.
        Example: python [PATH]/update_iam_simulation.py \\ 
            update_advisor_iam_role sarmad advisor/jonathan
        """,
    )
    parser_advisor_roles.set_defaults(func=update_advisor_iam_role)
    parser_advisor_roles.add_argument("client_name", type=str)
    parser_advisor_roles.add_argument("advisor_role_id", type=str)

    parser_10k_access = subparsers.add_parser(
        "update_10k_access",
        help="""
        Update (simulated) IAM access for 10ks.
        Each role has access to a specific list of tickers.
        Example: python [PATH]/update_iam_simulation.py \\
            update_10k_access advisor/jonathan AAPL,MSFT
        """,
    )

    parser_10k_access.set_defaults(func=update_10k_access)
    parser_10k_access.add_argument("advisor_role_id", type=str)
    parser_10k_access.add_argument("tickers", type=str)

    args = parser.parse_args()
    return args.func(args)


def update_10k_access(args: argparse.Namespace):
    path = "python/src/semantic_retrieval/examples/financial_report/access_control/iam_simulation_db.json"
    db_iam_simulation = json.loads(file_contents(path))
    access_10ks = db_iam_simulation["access_10ks"]
    access_10ks[args.advisor_role_id] = [t.strip() for t in args.tickers.split(",")]
    db_iam_simulation["access_10ks"] = access_10ks
    out = text_file_write(path, json.dumps(db_iam_simulation, indent=2))
    print(f"Updated 10k access: {args.advisor_role_id} -> {args.tickers}")
    return out


def update_advisor_iam_role(args: argparse.Namespace):
    path = "python/src/semantic_retrieval/examples/financial_report/access_control/iam_simulation_db.json"
    db_iam_simulation = json.loads(file_contents(path))
    advisors = db_iam_simulation["advisors"]
    advisors[args.client_name] = args.advisor_role_id
    db_iam_simulation["advisors"] = advisors
    out = text_file_write(path, json.dumps(db_iam_simulation, indent=2))
    print(
        f"Updated IAM: client: {args.client_name}, advisor role: {args.advisor_role_id}"
    )
    return out


if __name__ == "__main__":
    sys.exit(main())
