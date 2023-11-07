import json
import os
import re
from result import Err, Ok
from semantic_retrieval.common.core import file_contents
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB


async def validate_portfolio_access(resource_auth_id: str, viewer_auth_id: str) -> bool:
    # In production, this function could do an IAM lookup, DB access, etc.
    # For this simulation, we read from a local JSON file.

    # In this case, the resource_auth_id is the csv path
    # and the viewer_auth_id is the advisor name.

    if "admin" in viewer_auth_id:
        return True

    basename = os.path.basename(resource_auth_id)
    re_client_name = re.search(r"(.*)_portfolio.csv", basename)
    if not re_client_name:
        return False

    client_name = re_client_name.groups()[0]

    # User can look this up in real DB.
    path = "python/src/semantic_retrieval/examples/financial_report/access_control/iam_simulation_db.json"
    db_iam_simulation = json.loads(file_contents(path))["advisors"]

    return db_iam_simulation.get(client_name, None) == viewer_auth_id


async def validate_10k_access(
    resource_auth_id: str, viewer_auth_id: str, metadata_db: DocumentMetadataDB
) -> bool:
    if "admin" in viewer_auth_id:
        return True

    def _validate_10k_access_with_metadata(
        resource_auth_id: str,
        viewer_auth_id: str,
        metadata: DocumentMetadata,
    ):
        uri = metadata.uri
        ticker_re = re.search(r".*_([A-Z\.]+)\..*", uri)
        if not ticker_re:
            return False
        ticker = str(ticker_re.groups()[0])

        path = "python/src/semantic_retrieval/examples/financial_report/access_control/iam_simulation_db.json"
        db_iam_simulation = json.loads(file_contents(path))["access_10ks"]
        return ticker in db_iam_simulation.get(viewer_auth_id, [])

    res_metadata = await metadata_db.get_metadata(resource_auth_id)

    match res_metadata:
        case Err(_msg):
            # TODO [P1] log
            return False
        case Ok(metadata):
            return _validate_10k_access_with_metadata(
                metadata=metadata,
                viewer_auth_id=viewer_auth_id,
                resource_auth_id=resource_auth_id,
            )
