import { DocumentAccessPolicyFactory } from "../../access-control/documentAccessPolicyFactory";
import { Document, RawDocument } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import { ParserRegistry } from "./parserRegistry";

type ParserConfig = {
  metadataDB?: DocumentMetadataDB;
  accessControlPolicyFactory?: DocumentAccessPolicyFactory;
  parserRegistry?: ParserRegistry;
};

export async function parseDocuments(
  rawDocuments: RawDocument[],
  config: ParserConfig
): Promise<Document[]> {
  const parserRegistry = config.parserRegistry ?? new ParserRegistry();

  // Sanity check all parsers are available
  for (const rawDoc of rawDocuments) {
    if (!parserRegistry.getParser(rawDoc.mimeType)) {
      throw new Error("No parser available for MIME type: " + rawDoc.mimeType);
    }
  }

  return await Promise.all(
    rawDocuments.map(async (rawDocument) => {
      const document = await parserRegistry
        .getParser(rawDocument.mimeType)!
        .parse(rawDocument);

      if (config.metadataDB) {
        let accessPolicies;
        if (config.accessControlPolicyFactory) {
          accessPolicies =
            await config.accessControlPolicyFactory.getAccessPolicies(
              rawDocument
            );
        }

        await config.metadataDB.setMetadata(document.documentId, {
          documentId: document.documentId,
          rawDocument,
          document,
          uri: rawDocument.uri,
          name: rawDocument.name,
          mimeType: rawDocument.mimeType,
          metadata: {},
          attributes: {},
          accessPolicies,
        });
      }

      return document;
    })
  );
}
