import { VectorDB, VectorDBTextQuery } from "../data-store/vector-DBs.ts/vectorDB";
import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";

export type VectorDBRetrieverParams<V extends VectorDB> = {
    vectorDB: V;
    metadataDB?: DocumentMetadataDB;
    k?: number;
}

/**
 * Abstract class for retrieving Documents from and underlying VectorDB
 */
export class VectorDBRetriever<V extends VectorDB> extends BaseRetriever {
    vectorDB: V;

    constructor(params: VectorDBRetrieverParams<V>) {
        super(params.metadataDB);
        this.vectorDB = params.vectorDB;
    }

    protected async _getDocumentsUnsafe(params: BaseRetrieverQueryParams): Promise<Document[]> {
        const query: VectorDBTextQuery = {text: params.query};
        return await this.vectorDB.query(query);
    }
}