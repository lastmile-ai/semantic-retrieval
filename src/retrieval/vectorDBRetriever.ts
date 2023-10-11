import { VectorDB, VectorDBQuery } from "../data-store/vector-DBs/vectorDB";
import {
  Document,
  DocumentFragment,
  DocumentFragmentType,
} from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";
import { Md5 } from "ts-md5";
import { promises as fs } from "fs";

export type VectorDBRetrieverParams<V extends VectorDB> = {
  vectorDB: V;
  metadataDB: DocumentMetadataDB;
};

export interface VectorDBRetrieverQueryParams
  extends BaseRetrieverQueryParams<VectorDBQuery> {}

/**
 * Abstract class for retrieving data R from an underlying VectorDB
 */
export abstract class BaseVectorDBRetriever<
  V extends VectorDB,
  R,
> extends BaseRetriever<R> {
  vectorDB: V;

  constructor(params: VectorDBRetrieverParams<V>) {
    super(params.metadataDB);
    this.vectorDB = params.vectorDB;
  }

  protected async getDocumentsUnsafe(
    params: VectorDBRetrieverQueryParams
  ): Promise<Document[]> {
    const embeddings = await this.vectorDB.query(params.query);
    const fragments: (DocumentFragment | null)[] = await Promise.all(
      embeddings.map(async (embedding) => {
        const { documentId, fragmentId, ...embeddingMetadata } =
          embedding.metadata;

        if (!(fragmentId && documentId)) {
          console.error(
            "Retrieved embedding does not have fragmentId or documentId"
          );
          return null;
        }

        return {
          fragmentId,
          fragmentType: (embeddingMetadata.fragmentType ??
            "text") as DocumentFragmentType,
          documentId,
          metadata: embeddingMetadata,
          attributes: embedding.attributes,
          hash: Md5.hashStr(embedding.text),
          getContent: async () => embedding.text,
          serialize: async () => embedding.text,
        };
      })
    );

    const fragmentsByDocumentId = fragments.reduce(
      (acc, fragment) => {
        if (!fragment) {
          return acc;
        }
        if (acc[fragment.documentId] == null) {
          acc[fragment.documentId] = [];
        }
        acc[fragment.documentId].push(fragment);
        return acc;
      },
      {} as Record<string, DocumentFragment[]>
    );

    // We construct Document objects from the subset of fragments obtained from the query.
    // If needed, all Document fragments can be retrieved from the DocumentMetadataDB.
    return await Promise.all(
      Object.entries(fragmentsByDocumentId).map(
        async ([documentId, fragments]) => {
          const documentMetadata =
            await this.metadataDB.getMetadata(documentId);
          const storedDocument = documentMetadata.document;
          delete documentMetadata.document;
          return {
            ...documentMetadata,
            documentId,
            fragments,
            serialize: async () => {
              if (storedDocument) {
                return await storedDocument.serialize();
              }

              const serializedFragments = (
                await Promise.all(
                  fragments.map(async (fragment) => await fragment.serialize())
                )
              ).join("\n");

              const filePath = `${documentId}.txt`;
              await fs.writeFile(filePath, serializedFragments);
              return filePath;
            },
          };
        }
      )
    );
  }
}
