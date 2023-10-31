import { JSONObject } from "../../../src/common/jsonTypes";
import { VectorDBTextQuery } from "../../../src/data-store/vector-DBs/vectorDB";
import { Document } from "../../../src/document/document";
import { DocumentMetadataDB } from "../../../src/document/metadata/documentMetadataDB";
import { CSVRetriever } from "../../../src/retrieval/csvRetriever";
import { DocumentRetriever } from "../../../src/retrieval/documentRetriever";
import {
  BaseRetriever,
  BaseRetrieverQueryParams,
} from "../../../src/retrieval/retriever";
import { CallbackManager } from "../../../src/utils/callbacks";

export type FinancialReportData = {
  company: string;
  details: string;
}[];

type FinancialReportQuery = string;

export interface PortfolioData extends JSONObject {
  [Company: string]: { Shares: number | null };
}

export type FinancialReportDocumentRetrieverConfig = {
  metadataDB: DocumentMetadataDB;
  documentRetriever: DocumentRetriever<Document[], VectorDBTextQuery>;
  portfolioRetriever: CSVRetriever<PortfolioData>;
  callbackManager?: CallbackManager;
};

export class FinancialReportDocumentRetriever
  extends BaseRetriever<FinancialReportData, FinancialReportQuery>
  implements FinancialReportDocumentRetrieverConfig
{
  documentRetriever: DocumentRetriever<Document[], VectorDBTextQuery>;
  portfolioRetriever: CSVRetriever<PortfolioData>;
  metadataDB: DocumentMetadataDB;

  constructor(config: FinancialReportDocumentRetrieverConfig) {
    super(config.metadataDB, config.callbackManager);
    this.metadataDB = config.metadataDB;
    this.documentRetriever = config.documentRetriever;
    this.portfolioRetriever = config.portfolioRetriever;
  }

  // For each of the companies with owned shares in the Portfolio, retrieve embedded documents
  // related to the query from pinecone, and return the details for each company
  async retrieveData(
    params: BaseRetrieverQueryParams<FinancialReportQuery>
  ): Promise<FinancialReportData> {
    const portfolio = await this.portfolioRetriever.retrieveData({
      query: { primaryKeyColumn: "Company" },
    });

    const ownedCompanies = Object.keys(portfolio).filter(
      (company) => (portfolio[company].Shares ?? 0) > 0
    );

    const ownedCompaniesDocumentIds = await Promise.all(
      ownedCompanies.map(async (company) => ({
        company,
        documentIds: await this.metadataDB.queryDocumentIds({
          type: "string_field",
          fieldName: "name",
          fieldValue: company,
          matchType: "includes",
        }),
      }))
    );

    console.log("ownedCompaniesDocumentIds", ownedCompaniesDocumentIds);

    const reportDocuments = await Promise.all(
      ownedCompaniesDocumentIds.map(async ({ company, documentIds }) => {
        const documents = (
          await Promise.all(
            documentIds.map((documentId) =>
              this.documentRetriever.retrieveData({
                query: {
                  text: params.query,
                  topK: 5,
                  metadataFilter: {
                    documentId,
                  },
                },
              })
            )
          )
        ).flat();

        return {
          company,
          documents,
        };
      })
    );

    return await Promise.all(
      reportDocuments.map(async (report) => {
        const details = (
          await Promise.all(
            report.documents.map((document) =>
              document.fragments.map((fragment) => fragment.getContent())
            )
          )
        )
          .flat()
          .join("\n");

        return {
          company: report.company,
          details,
        };
      })
    );
  }
}
