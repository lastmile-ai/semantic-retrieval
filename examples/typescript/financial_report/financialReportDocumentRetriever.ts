import { DocumentFragment, Document } from "../../../src/document/document";
import { DocumentMetadataDB } from "../../../src/document/metadata/documentMetadataDB";
import { CSVRetriever } from "../../../src/retrieval/csvRetriever";
import { DocumentRetriever } from "../../../src/retrieval/documentRetriever";
import { BaseRetrieverQueryParams } from "../../../src/retrieval/retriever";
import { CallbackManager } from "../../../src/utils/callbacks";

export type FinancialReportData = {
  company: string;
  details: string;
}[];

export type FinancialReportQuery = string;

type PortfolioData = {
  [Company: string]: { Shares: number | null };
};

export type FinancialReportDocumentRetrieverConfig = {
  metadataDB: DocumentMetadataDB;
  documentRetriever: DocumentRetriever<Document, FinancialReportQuery>;
  portfolioRetriever: CSVRetriever<PortfolioData>;
  callbackManager?: CallbackManager;
};

export class FinancialReportDocumentRetriever extends DocumentRetriever<
  FinancialReportData,
  FinancialReportQuery
> {
  documentRetriever: DocumentRetriever<Document, FinancialReportQuery>;
  portfolioRetriever: CSVRetriever<PortfolioData>;

  constructor(config: FinancialReportDocumentRetrieverConfig) {
    super(config.metadataDB, config.callbackManager);
    this.documentRetriever = config.documentRetriever;
    this.portfolioRetriever = config.portfolioRetriever;
  }

  protected getFragmentsUnsafe(
    params: BaseRetrieverQueryParams<string>
  ): Promise<DocumentFragment[]> {
    // For each of the companies with owned shares in the Portfolio, retrieve the DocumentFragments
    // relevant to the query

    const portfolio = this.portfolioRetriever.retrieveData({
      query: { primaryKeyColumn: "Company" },
    });
    const ownedCompanies = Object.keys(portfolio).filter(
      (company) => portfolio[company].Shares != null
    );

    const ownedCompaniesDocumentIds = ownedCompanies.map((company) => {
      return this.metadataDB.queryDocumentIds({
        metadataKey: "source",
        metadataValue: company,
        matchType: "includes",
      });
    });
  }

  protected processDocuments(
    _documents: Document[]
  ): Promise<FinancialReportData> {
    throw new Error("Method not implemented.");
  }
}
