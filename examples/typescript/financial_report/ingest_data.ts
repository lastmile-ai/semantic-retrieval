#!/usr/bin/env ts-node
// Description: Script to ingest 10k financial report documents into a Pinecone index and serialize associated metadata to disk
// Usage Example: npx ts-node examples/typescript/financial_report/ingest_data.ts -p my-index

import { program } from "commander";
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

program
  .name("ingest_data")
  .description(
    "Script to ingest 10k financial report documents into a Pinecone index and serialize associated metadata to disk"
  );

program.option(
  "-p, --pinecone_index [PINECONE_INDEX]",
  // Make sure this matches your Pinecone index name & it has 1536 dimensions for openai embeddings
  "specify the name of the pinecone index to ingest the documents into",
  "test-financial-report" // default pinecone index name
);

program.parse(process.argv);

async function main() {
  console.log("Starting ingestion...");

  const options = program.opts();
  const { indexName } = options;
  if (typeof indexName !== "string") {
    throw new Error("no index name or default specified");
  }

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

  console.log(
    `Transformed ${transformedDocuments.length} documents for ingestion`
  );

  // Persist metadataDB to disk for loading in the other scripts
  await metadataDB.persist(
    "examples/typescript/financial_report/metadataDB.json"
  );

  console.log("Persisted metadataDB to disk");

  // Use a new namespace for each run so that we can easily change which data to use
  const namespace = uuid();
  console.log(`NAMESPACE: ${namespace}`);

  await PineconeVectorDB.fromDocuments(transformedDocuments, {
    indexName,
    namespace,
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });

  console.log("Ingestion complete");
}

main()
  .then(() => {
    console.log("Done!");
  })
  .catch((err: any) => {
    console.error("Error:", err);
    process.exit(1);
  })
  .finally(() => {
    process.exit(0);
  });

export {};
