// TODO: These imports should be from actual lastmile retrieval package
import { AccessPassport } from "../src/access-control/accessPassport";
import { AlwaysAllowDocumentAccessPolicyFactory } from "../src/access-control/alwaysAllowDocumentAccessPolicyFactory";
import { PineconeVectorDB } from "../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../src/document/metadata/InMemoryDocumentMetadataDB";
import { FileSystem } from "../src/ingestion/data-sources/dataSource";
import * as SimpleDocumentParser from "../src/ingestion/document-parsers/simpleDocumentParser";
import { OpenAICompletionGenerator } from "../src/generator/llm/openAICompletionGenerator";
import { VectorDBDocumentRetriever } from "../src/retrieval/vectorDBDocumentRetriever";

const metadataDB = new InMemoryDocumentMetadataDB();

async function createIndex() {
  const fileSystem = new FileSystem("./example_docs");
  const rawDocuments = await fileSystem.loadDocuments();

  const parsedDocuments = await SimpleDocumentParser.parseDocuments(
    rawDocuments,
    {
      metadataDB,
      accessControlPolicyFactory: new AlwaysAllowDocumentAccessPolicyFactory(),
    },
  );
  return await PineconeVectorDB.fromDocuments(parsedDocuments, metadataDB);
}

async function main() {
  const vectorDB = await createIndex();
  const accessPassport = new AccessPassport();
  const retriever = new VectorDBDocumentRetriever(vectorDB);
  const generator = new OpenAICompletionGenerator();
  const res = await generator.run({
    accessPassport,
    prompt: "How do I use parameters in a workbook?",
    retriever,
  });
  console.log(res);
}

main();

export {};
