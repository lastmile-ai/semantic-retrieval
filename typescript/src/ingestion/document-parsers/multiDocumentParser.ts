import { DocumentAccessPolicyFactory } from "../../access-control/documentAccessPolicyFactory";
import { IngestedDocument, RawDocument } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import { CallbackManager } from "../../utils/callbacks";
import { ParserRegistry } from "./parserRegistry";

type ParserConfig = {
  metadataDB?: DocumentMetadataDB;
  accessControlPolicyFactory?: DocumentAccessPolicyFactory;
  parserRegistry?: ParserRegistry;
  callbackManager?: CallbackManager;
};

/**
 * A basic implementation for parsing Documents from RawDocuments, using
 * a config to determine which specific parsers to use for each MIME type.
 * If a metadataDB and accessControlPolicyFactory are provided, the parsed
 * documents will have relevant metadata and access policies set in the metadataDB.
 * @param rawDocuments An array of RawDocuments to parse into Documents
 * @param config A ParserConfig for configuring the parsing implementation
 * @returns
 */
export async function parseDocuments(
  rawDocuments: RawDocument[],
  config: ParserConfig
): Promise<IngestedDocument[]> {
  const parserRegistry = config.parserRegistry ?? new ParserRegistry(config);

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
          hash: rawDocument.hash,
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
