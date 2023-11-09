#!/usr/bin/env ts-node
// Description: Script to ingest 10k financial report documents into a Pinecone index and serialize associated metadata to disk
// Usage Example: npx ts-node examples/typescript/financial_report/ingest_data.ts -p my-index

import { OptionValues, program } from "commander";
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";
import * as MultiDocumentParser from "../../src/ingestion/document-parsers/multiDocumentParser";
import dotenv from "dotenv";
import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";
import { AlwaysAllowDocumentAccessPolicyFactory } from "../../src/access-control/alwaysAllowDocumentAccessPolicyFactory";
import { SeparatorTextChunker } from "../../src/transformation/document/text/separatorTextChunker";
import { PineconeVectorDB } from "../../src/data-store/vector-DBs/pineconeVectorDB";
import { OpenAIEmbeddings } from "../../src/transformation/embeddings/openAIEmbeddings";
import { v4 as uuid } from "uuid";
import {
  AddDocumentsToVectorDBEvent,
  CallbackManager,
  LoadChunkedContentEvent,
  LoadDocumentsSuccessEvent,
  ParseSuccessEvent,
  TransformDocumentEvent,
  TransformDocumentsEvent,
} from "../../../typescript/src/utils/callbacks";

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
  "examples" // default pinecone index name
);

program.option("-v, --verbose", "specify whether to print verbose logs", false);

program.parse(process.argv);

async function main() {
  console.log("Starting ingestion...");

  const options = program.opts();
  const { indexName, verboseLogging } = getOptions(options);

  const callbackManager = getLoggingCallbackManager(verboseLogging);

  const fileSystem = new FileSystem({
    path: "../examples/example_data/financial_report/10ks",
    callbackManager,
  });
  const rawDocuments = await fileSystem.loadDocuments();
  const metadataDB = new InMemoryDocumentMetadataDB();

  const parsedDocuments = await MultiDocumentParser.parseDocuments(
    rawDocuments,
    {
      metadataDB,
      // Always allow by default. Use update_access_control script to change
      accessControlPolicyFactory: new AlwaysAllowDocumentAccessPolicyFactory(),
      callbackManager,
    }
  );

  const documentTransformer = new SeparatorTextChunker({
    metadataDB,
    callbackManager,
  });

  const transformedDocuments =
    await documentTransformer.transformDocuments(parsedDocuments);

  // Persist metadataDB to disk for loading in the other scripts
  await metadataDB.persist("examples/financial_report/metadataDB.json");

  console.log("Persisted metadataDB to disk");

  // Use a new namespace for each run so that we can easily change which data to use
  const namespace = uuid();
  console.log(`NAMESPACE: ${namespace}`);

  const vectorDB = new PineconeVectorDB({
    indexName,
    namespace,
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
    callbackManager,
  });

  console.log("Upserting document embeddings to Pinecone index...");

  await vectorDB.addDocuments(transformedDocuments);

  console.log(`Ingestion to namespace ${namespace} complete!}`);
}

function getLoggingCallbackManager(verboseLogging: boolean) {
  return new CallbackManager(`ingest-data-${uuid()}`, {
    onLoadChunkedContent: [
      async (event: LoadChunkedContentEvent) => {
        if (verboseLogging) {
          console.log(
            `Loaded ${event.chunkedContent.length} chunks from ${event.path} via ${event.loader.constructor.name}: `,
            event.chunkedContent
          );
        } else {
          console.log(
            `Loaded ${event.chunkedContent.length} chunks from ${event.path} via ${event.loader.constructor.name}`
          );
        }
      },
    ],
    onLoadDocumentsSuccess: [
      async (event: LoadDocumentsSuccessEvent) => {
        if (verboseLogging) {
          console.log(
            `Successfully loaded ${event.rawDocuments.length} documents from ${event.dataSource.name}: `,
            event.dataSource,
            event.rawDocuments
          );
        } else {
          console.log(
            `Successfully loaded ${event.rawDocuments.length} documents from ${event.dataSource.name}`
          );
        }
      },
    ],
    onParseSuccess: [
      async (event: ParseSuccessEvent) => {
        if (verboseLogging) {
          console.log(
            `Parsed ${event.ingestedDocument.rawDocument.uri}: `,
            event.ingestedDocument
          );
        } else {
          console.log(
            `Parsed ${event.ingestedDocument.rawDocument.uri} into ${event.ingestedDocument.fragments.length} fragments`
          );
        }
      },
    ],
    onTransformDocument: [
      async (event: TransformDocumentEvent) => {
        if (verboseLogging) {
          console.log(
            `Transformed document ${event.originalDocument.documentId} to ${event.transformedDocument.documentId}`,
            event.originalDocument,
            event.transformedDocument
          );
        } else {
          console.log(
            `Transformed document ${event.originalDocument.documentId} to ${event.transformedDocument.documentId}`
          );
        }
      },
    ],
    onTransformDocuments: [
      async (event: TransformDocumentsEvent) => {
        console.log(
          `Transformed ${event.originalDocuments.length} documents for ingestion`
        );
      },
    ],
    onAddDocumentsToVectorDB: [
      async (event: AddDocumentsToVectorDBEvent) => {
        console.log(`Upserted ${event.documents.length} documents to Pinecone`);
      },
    ],
  });
}

function getOptions(options: OptionValues) {
  const { pinecone_index: indexName, verbose } = options;

  if (typeof indexName !== "string") {
    throw new Error("no index name or default specified");
  }

  return {
    indexName,
    verboseLogging: verbose !== false,
  };
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
