import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbeddings";

async function main() {
  // TODO: This needs to be a metadata DB implementation that persists/loads from disk
  const metadataDB = new InMemoryDocumentMetadataDB();

  const _vectorDB = await new PineconeVectorDB({
    indexName: "test-financial-report",
    namespace: "GET NAMESPACE FROM ingest_data RUN",
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });

  // const retriever = new FinancialReportDocumentRetriever({documentRetriever: vectorDB, metadataDB});
  // const generator = new FinancialReportGenerator(new OpenAIChatModel());

  // const accessPassport = new AccessPassport();
  // accessPassport.registerIdentity("DemoAccess", "User" or "Admin");

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
