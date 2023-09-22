import { VectorDB, VectorDBTextQuery } from "../data-store/vector-DBs.ts/vectorDB";
import { Document } from "../document/document";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";

export type VectorDBRetrieverParams<V extends VectorDB> = {
    vectorDB: V;
    k?: number;
}

/**
 * Abstract class for retrieving Documents from and underlying VectorDB
 */
export class VectorDBRetriever<V extends VectorDB> extends BaseRetriever {
    vectorDB: V;

    constructor(params: VectorDBRetrieverParams<V>) {
        super();
        this.vectorDB = params.vectorDB;
    }

    protected async _getDocumentsUnsafe(params: BaseRetrieverQueryParams): Promise<Document[]> {
        const query: VectorDBTextQuery = {text: params.query};
        return await this.vectorDB.query(query);
    }
}