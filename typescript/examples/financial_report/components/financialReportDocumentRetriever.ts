import { JSONObject } from "../../../src/common/jsonTypes";
import { DocumentMetadataDB } from "../../../src/document/metadata/documentMetadataDB";
import { CSVRetriever } from "../../../src/retrieval/csvRetriever";
import {
  BaseRetriever,
  BaseRetrieverQueryParams,
} from "../../../src/retrieval/retriever";
import { VectorDBDocumentRetriever } from "../../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import {
  CallbackManager,
  RetrieveDataEvent,
} from "../../../src/utils/callbacks";

export type FinancialReportData = {
  company: string;
  profile: string;
  details: string;
}[];

type FinancialReportQuery = BaseRetrieverQueryParams<string>;

export interface PortfolioData extends JSONObject {
  [Company: string]: { Shares: number | null };
}

export interface CompanyProfiles extends JSONObject {
  [Company: string]: { Profile: string };
}

export type FinancialReportDocumentRetrieverConfig = {
  metadataDB: DocumentMetadataDB;
  documentRetriever: VectorDBDocumentRetriever;
  portfolioRetriever: CSVRetriever<PortfolioData>;
  companyProfilesRetriever: CSVRetriever<CompanyProfiles>;
  callbackManager?: CallbackManager;
};

export class FinancialReportDocumentRetriever
  extends BaseRetriever
  implements FinancialReportDocumentRetrieverConfig
{
  documentRetriever: VectorDBDocumentRetriever;
  portfolioRetriever: CSVRetriever<PortfolioData>;
  companyProfilesRetriever: CSVRetriever<CompanyProfiles>;
  metadataDB: DocumentMetadataDB;

  constructor(config: FinancialReportDocumentRetrieverConfig) {
    super(config.metadataDB, config.callbackManager);
    this.metadataDB = config.metadataDB;
    this.documentRetriever = config.documentRetriever;
    this.portfolioRetriever = config.portfolioRetriever;
    this.companyProfilesRetriever = config.companyProfilesRetriever;
  }

  // For each of the companies with owned shares in the Portfolio, retrieve embedded documents
  // related to the query from pinecone, and return the details for each company
  async retrieveData(
    params: FinancialReportQuery
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
          fieldName: "uri",
          fieldValue: company,
          matchType: "includes",
        }),
      }))
    );

    const reportDocuments = await Promise.all(
      ownedCompaniesDocumentIds.map(async ({ company, documentIds }) => {
        const documents = (
          await Promise.all(
            documentIds.map((documentId) =>
              this.documentRetriever.retrieveData({
                accessPassport: params.accessPassport,
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

    const companyProfiles = await this.companyProfilesRetriever.retrieveData({
      query: { primaryKeyColumn: "Company" },
    });

    const data = await Promise.all(
      reportDocuments.map(async (report) => {
        const fragmentContentPromises: Promise<string>[] = [];

        report.documents.forEach((document) => {
          document.fragments.forEach((fragment) => {
            fragmentContentPromises.push(fragment.getContent());
          });
        });

        const details = (await Promise.all(fragmentContentPromises))
          .flat()
          .join("\n");

        return {
          company: report.company,
          profile: companyProfiles[report.company].Profile,
          details,
        };
      })
    );

    const event: RetrieveDataEvent = {
      name: "onRetrieveData",
      params,
      data,
    };

    await this.callbackManager?.runCallbacks(event);

    return data;
  }
}
