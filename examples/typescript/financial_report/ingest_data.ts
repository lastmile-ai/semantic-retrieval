import { FileSystem } from "../../../src/ingestion/data-sources/fs/fileSystem";
import * as MultiDocumentParser from "../../../src/ingestion/document-parsers/multiDocumentParser";
import dotenv from "dotenv";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import { AlwaysAllowDocumentAccessPolicyFactory } from "../../../src/access-control/alwaysAllowDocumentAccessPolicyFactory";
import { SeparatorTextChunker } from "../../../src/transformation/document/text/separatorTextChunker";
import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbeddings";
import { v4 as uuid } from "uuid";

dotenv.config();

async function main() {
  const fileSystem = new FileSystem(
    "examples/example_data/financial_report/10ks"
  );
  const rawDocuments = await fileSystem.loadDocuments();
  const metadataDB = new InMemoryDocumentMetadataDB();

  const parsedDocuments = await MultiDocumentParser.parseDocuments(
    rawDocuments,
    {
      metadataDB,
      // Always allow by default. Use update_access_control script to change
      accessControlPolicyFactory: new AlwaysAllowDocumentAccessPolicyFactory(),
    }
  );

  const documentTransformer = new SeparatorTextChunker({ metadataDB });
  const transformedDocuments =
    await documentTransformer.transformDocuments(parsedDocuments);

  // Persist metadataDB to disk for loading in the other scripts
  await metadataDB.persist(
    "examples/typescript/financial_report/metadataDB.json"
  );

  // Use a new namespace for each run so that we can easily change which data to use
  const namespace = uuid();
  console.log(`NAMESPACE: ${namespace}`);

  await PineconeVectorDB.fromDocuments(transformedDocuments, {
    // Make sure this matches your Pinecone index name & it has 1536 dimensions for openai embeddings
    indexName: "test-financial-report",
    namespace,
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });

  console.log("Ingestion complete");
}

main();

export {};
