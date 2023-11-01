import { AccessPassport } from "../../../src/access-control/accessPassport";
import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import { CSVRetriever } from "../../../src/retrieval/csvRetriever";
import { VectorDBDocumentRetriever } from "../../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbeddings";
import { AdvisorIdentity } from "./components/access_control/advisorIdentity";
import {
  FinancialReportDocumentRetriever,
  PortfolioData,
} from "./components/financialReportDocumentRetriever";
import dotenv from "dotenv";
import { ResourceAccessPolicy } from "../../../src/access-control/resourceAccessPolicy";
import { AlwaysAllowAccessPolicy } from "../../../src/access-control/policies/alwaysAllowAccessPolicy";
import fs from "fs/promises";
import { FinancialReportGenerator } from "./components/financialReportGenerator";
import { SecretReportAccessPolicy } from "./components/access_control/secretReportAccessPolicy";

dotenv.config();

async function main() {
  // Load the metadataDB persisted from ingest_data script
  const metadataDB = await InMemoryDocumentMetadataDB.fromJSONFile(
    "examples/typescript/financial_report/metadataDB.json",
    (key, value) => {
      if (key === "accessPolicies") {
        // deserialize access policies to their instances
        return (value as ResourceAccessPolicy[]).map(
          (policy: ResourceAccessPolicy) => {
            switch (policy.policy) {
              case "AlwaysAllowAccessPolicy":
                return new AlwaysAllowAccessPolicy();
              case "SecretReportAccessPolicy":
                return new SecretReportAccessPolicy();
              default:
                return policy;
            }
          }
        );
      }
      return value;
    }
  );

  const vectorDB = await new PineconeVectorDB({
    indexName: "test-financial-report",
    // TODO: Make this dynamic via script param
    namespace: "ea4bcf44-e0f3-46ff-bf66-5b1f9e7502df",
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });

  const documentRetriever = new VectorDBDocumentRetriever({
    vectorDB,
    metadataDB,
  });

  // TODO: Make this dynamic via script param
  const portfolioRetriever = new CSVRetriever<PortfolioData>(
    "examples/example_data/financial_report/portfolios/client_a_portfolio.csv"
  );

  const accessPassport = new AccessPassport();
  const identity = new AdvisorIdentity(); // TODO: Make this dynamic via script param
  accessPassport.register(identity);

  const retriever = new FinancialReportDocumentRetriever({
    documentRetriever,
    portfolioRetriever,
    metadataDB,
  });

  const generator = new FinancialReportGenerator();

  const report = await generator.run({
    prompt: "overall cash flow",
    structure:
      "Paragraphs delineated by heading containing company name and symbol",
    accessPassport,
    retriever,
  });

  await fs.writeFile("examples/typescript/financial_report/report.txt", report);
}

main();

export {};
