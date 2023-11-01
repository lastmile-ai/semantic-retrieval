import { AccessPassport } from "../../../src/access-control/accessPassport";
import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import { OpenAIChatModel } from "../../../src/generator/completion-models/openai/openAIChatModel";
import { CSVRetriever } from "../../../src/retrieval/csvRetriever";
import { VectorDBDocumentRetriever } from "../../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbeddings";
import { AdvisorIdentity } from "./access_control/advisorIdentity";
import {
  FinancialReportDocumentRetriever,
  PortfolioData,
} from "./financialReportDocumentRetriever";
import dotenv from "dotenv";
import { ResourceAccessPolicy } from "../../../src/access-control/resourceAccessPolicy";
import { AlwaysAllowAccessPolicy } from "../../../src/access-control/policies/alwaysAllowAccessPolicy";
import fs from "fs/promises";
import { FinancialReportGenerator } from "./financialReportGenerator";

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
            if (policy.policy === "AlwaysAllowAccessPolicy") {
              return new AlwaysAllowAccessPolicy();
            }
            return policy;
            // TODO: Handle other policies for demo. Could also try to dynamically import from
            // policies dir based on name, or have static mapping of policy => class constructor
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
  // const identity = new AdvisorIdentity("client_a"); // TODO: Make this dynamic via script param
  // accessPassport.register(identity);

  const retriever = new FinancialReportDocumentRetriever({
    documentRetriever,
    portfolioRetriever,
    metadataDB,
  });

  const generator = new FinancialReportGenerator();

  const report = await generator.run({
    prompt: "Recovery from the COVIDE-19 pandemic",
    accessPassport, // not necessary in this case, but include for example
    retriever,
  });

  await fs.writeFile("examples/typescript/financial_report/report.txt", report);
}

main();

export {};
