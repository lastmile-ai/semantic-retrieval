// TODO: These imports should be from actual lastmile retrieval package
import { AccessPassport } from "../../src/access-control/accessPassport";
import { AlwaysAllowDocumentAccessPolicyFactory } from "../../src/access-control/alwaysAllowDocumentAccessPolicyFactory";
import { PineconeVectorDB } from "../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/InMemoryDocumentMetadataDB";
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";
import * as MultiDocumentParser from "../../src/ingestion/document-parsers/multiDocumentParser";
import { OpenAIChatModel } from "../../src/generator/completion-models/openai/openAIChatModel";
import { VectorDBDocumentRetriever } from "../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { SeparatorTextChunker } from "../../src/transformation/document/text/separatorTextChunker";
import { OpenAIEmbeddings } from "../../src/transformation/embeddings/openAIEmbeddings";
import { VectorDBRAGCompletionGenerator } from "../../src/generator/retrieval-augmented-generation/vectorDBRAGCompletionGenerator";
import dotenv from "dotenv";

dotenv.config();

const metadataDB = new InMemoryDocumentMetadataDB();

async function createIndex() {
  const fileSystem = new FileSystem("examples/example_data");
  const rawDocuments = await fileSystem.loadDocuments();

  const parsedDocuments = await MultiDocumentParser.parseDocuments(
    rawDocuments,
    {
      metadataDB,
      accessControlPolicyFactory: new AlwaysAllowDocumentAccessPolicyFactory(),
    }
  );

  const documentTransformer = new SeparatorTextChunker({ metadataDB });
  const transformedDocuments =
    await documentTransformer.transformDocuments(parsedDocuments);

  return await PineconeVectorDB.fromDocuments(transformedDocuments, {
    indexName: "test-index",
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });
}

async function main() {
  const vectorDB = await createIndex();
  const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });
  const generator = new VectorDBRAGCompletionGenerator(new OpenAIChatModel());

  const res = await generator.run({
    accessPassport: new AccessPassport(), // not necessary in this case, but include for example
    prompt: "How do I use parameters in a workbook?",
    retriever,
  });
  console.log(res);
}

main();

export {};
