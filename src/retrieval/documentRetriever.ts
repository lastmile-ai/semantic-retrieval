import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { BaseRetriever } from "./retriever";

export abstract class BaseDocumentRetriever extends BaseRetriever<Document[]> {
  constructor(metadataDB?: DocumentMetadataDB) {
    super(metadataDB);
  }

  /**
   * Perform any post-processing on the retrieved Documents.
   * @param documents The array of retrieved Documents to post-process.
   * @returns A promise that resolves to post-processed data.
   */
  protected async _processDocuments(
    documents: Document[]
  ): Promise<Document[]> {
    return documents;
  }
}
