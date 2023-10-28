import { AccessPassport } from "../../../src/access-control/accessPassport";
import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import { OpenAIChatModel } from "../../../src/generator/completion-models/openai/openAIChatModel";
import { CSVRetriever } from "../../../src/retrieval/csvRetriever";
import { VectorDBDocumentRetriever } from "../../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbeddings";
import { AdvisorIdentity } from "./access_control/advisorIdentity";

async function main() {
  // Load the metadataDB persisted from ingest_data script
  const metadataDB = await InMemoryDocumentMetadataDB.fromJSONFile(
    "examples/typescript/financial_report/metadataDB.json"
  );

  const vectorDB = await new PineconeVectorDB({
    indexName: "test-financial-report",
    // TODO: Make this dynamic via script param
    namespace: "GET NAMESPACE FROM ingest_data RUN",
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });

  const _documentRetriever = new VectorDBDocumentRetriever({
    vectorDB,
    metadataDB,
  });

  // TODO: Make this dynamic via script param
  const _portfolioRetriever = new CSVRetriever(
    "examples/example_data/financial_report/portfolios/client_a_portfolio.csv"
  );

  const accessPassport = new AccessPassport();
  const identity = new AdvisorIdentity("client_a"); // TODO: Make this dynamic via script param
  accessPassport.register(identity);

  // const retriever = new FinancialReportDocumentRetriever({
  //   accessPassport,
  //   documentRetriever,
  //   portfolioRetriever,
  //   metadataDB,
  // });

  // const generator = new FinancialReportGenerator({
  //   model: new OpenAIChatModel(),
  //   retriever,
  // });

  // const prompt = new PromptTemplate("Use the following data to construct a financial report matching the following format ... {data}")

  //   const res = await generator.run({
  //     accessPassport, // not necessary in this case, but include for example
  //     prompt,
  //     retriever,
  //   });

  // TODO: Save res to disk
}

main();

export {};
