import logging
import re
from typing import Dict, List, NewType

from result import Err, Ok, Result
from semantic_retrieval.access_control.access_function import AccessFunction
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity
from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.common.types import Record
from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)
from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDBTextQuery
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.retrieval.csv_retriever import CSVRetriever
from semantic_retrieval.transformation.embeddings.embeddings import VectorEmbedding
from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable
from semantic_retrieval.utils.interop import canonical_field


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class FinancialReportData(Record):
    company: str
    details: str


PortfolioData = NewType("PortfolioData", Dict[str, int])


class FinancialReportDocumentRetriever(Traceable):
    vector_db: PineconeVectorDB
    portfolio_retriever: CSVRetriever
    metadata_db: DocumentMetadataDB
    callback_manager: CallbackManager

    def __init__(
        self,
        vector_db_config: PineconeVectorDBConfig,
        embeddings_config: OpenAIEmbeddingsConfig,
        portfolio: PortfolioData,
        metadata_db: DocumentMetadataDB,
        user_access_function: AccessFunction,
        viewer_identity: AuthenticatedIdentity,
        callback_manager: CallbackManager,
    ) -> None:
        embeddings = OpenAIEmbeddings(
            embeddings_config, callback_manager=callback_manager
        )

        self.viewer_identity = viewer_identity
        self.user_access_function = user_access_function

        self.portfolio = portfolio
        self.metadata_db = metadata_db
        self.callback_manager = callback_manager

        self.vector_db = PineconeVectorDB(
            vector_db_config,
            embeddings=embeddings,
            metadata_db=metadata_db,
            user_access_function=self.user_access_function,
            viewer_identity=self.viewer_identity,
            callback_manager=self.callback_manager,
        )

    async def retrieve_data(
        self,
        query: str,
        portfolio: PortfolioData,
        # TODO [P1] pull this stuff out into a Record
        top_k: int,
        overfetch_factor: float = 1.0,
    ) -> Result[List[FinancialReportData], str]:
        vectordb_text_query = VectorDBTextQuery(
            mode="text",
            topK=int(overfetch_factor * top_k),
            text=query,
            # TODO [P1]
            metadata_filter={},
        )

        knn = await self.vector_db.query(vectordb_text_query)

        def _get_doc_id(result: VectorEmbedding) -> str:
            logger.debug(f"{(result.metadata or {}).keys()=}")
            if result.metadata is None:
                return ""
            else:
                return result.metadata[canonical_field("document_id")]

        retrieved_doc_ids = {_get_doc_id(result) for result in knn}

        metadata = {
            doc_id: await self.metadata_db.get_metadata(doc_id)
            for doc_id in retrieved_doc_ids
        }

        out: List[FinancialReportData] = []
        for knn_result in knn:
            doc_id = _get_doc_id(knn_result)
            if doc_id not in metadata:
                # TODO [P1] log
                continue

            meta = metadata[doc_id]
            match meta:
                case Err(msg):
                    # TODO [P1] log
                    print(msg)
                    continue
                case Ok(DocumentMetadata(uri=uri)):
                    res_ticker = _uri_extract_ticker(uri)
                    match res_ticker:
                        case Err(msg):
                            print(f"error ticker result: {msg=}")
                        case Ok(ticker):
                            res_f_data = (
                                _get_financial_report_data_with_ticker_if_in_portfolio(
                                    portfolio, ticker, knn_result
                                )
                            )
                            match res_f_data:
                                case Err(msg):
                                    # Fine, not in portfolio, continue
                                    continue
                                case Ok(f_data):
                                    out.append(f_data)

        return Ok(out)


def _uri_extract_ticker(uri: str) -> Result[str, str]:
    try:
        return Ok(str(re.search(r"_([\w\.]+).md", uri).groups()[0]).upper())  # type: ignore [fixme]
    except Exception as e:
        return Err(f"exn,{e=}, {uri=}")


def _get_financial_report_data_with_ticker_if_in_portfolio(
    portfolio: PortfolioData, ticker: str, knn_result: VectorEmbedding
) -> Result[FinancialReportData, str]:
    if ticker in portfolio and portfolio[ticker] > 0:
        return Ok(
            FinancialReportData(
                company=ticker,
                details=knn_result.text,
            )
        )
    else:
        return Err(f"no shares for {ticker=}")
