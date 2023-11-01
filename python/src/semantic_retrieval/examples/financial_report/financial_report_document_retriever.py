import re
from typing import Dict, List, NewType

from result import Err, Ok, Result
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


class FinancialReportData(Record):
    company: str
    details: str


PortfolioData = NewType("PortfolioData", Dict[str, int])


class FinancialReportDocumentRetriever:
    vector_db: PineconeVectorDB
    portfolio_retriever: CSVRetriever[PortfolioData]
    metadata_db: DocumentMetadataDB

    def __init__(
        self,
        # access_passport: AccessPassport,
        vector_db_config: PineconeVectorDBConfig,
        embeddings_config: OpenAIEmbeddingsConfig,
        portfolio: CSVRetriever[PortfolioData],
        metadata_db: DocumentMetadataDB,
    ) -> None:
        embeddings = OpenAIEmbeddings(embeddings_config)

        self.vector_db = PineconeVectorDB(
            vector_db_config,
            embeddings=embeddings,
            metadata_db=metadata_db,
        )
        self.portfolio = portfolio
        self.metadata_db = metadata_db

    async def retrieve_data(
        self,
        query: str,
        portfolio: PortfolioData,
        # TODO pull this stuff out into a Record
        top_k: int,
        overfetch_factor: float = 1.0,
    ) -> List[FinancialReportData]:
        vdbq = VectorDBTextQuery(
            mode="text",
            topK=int(overfetch_factor * top_k),
            text=query,
            # TODO
            metadata_filter={},
        )

        knn = await self.vector_db.query(vdbq)

        def _get_doc_id(result: VectorEmbedding) -> str:
            if result.metadata is None:
                return ""
            else:
                return result.metadata["documentId"]

        retrieved_doc_ids = {_get_doc_id(result) for result in knn}

        metadata = {
            doc_id: await self.metadata_db.get_metadata(doc_id)
            for doc_id in retrieved_doc_ids
        }

        out: List[FinancialReportData] = []
        for knn_result in knn:
            doc_id = _get_doc_id(knn_result)
            if doc_id not in metadata:
                # TODO log
                continue

            meta = metadata[doc_id]
            match meta:
                case Err(msg):
                    # TODO log
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

        return out


def _uri_extract_ticker(uri: str):
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
