# semantic-retrieval

This project is under development.


# What is this library?
The LastMile semantic-retrieval library provides everything an enterprise should need to integrate powerful semantic retrieval flows into their systems.

# The Key Components of Semantic Retrieval

## Data Ingestion, Transformation & Indexing
Although LLM capabilities are continuously improving, there are still significant limitations to what can be achieved using LLMs on their own. Processes such retrieval-augmented generation (RAG) have had great success in lifting such limitations and producing high-quality LLM experiences. The key to such processes is the integration of external data into the LLM flow. Data can be ingested from a number of different sources, transformed into a format useful for the task at hand, and indexed in alternative data stores for optimal retrieval when it is needed.

The semantic-retrieval library has abstractions & concrete implementations for each step of this process:

### Data Ingestion

**Data Sources**
The `DataSource` class provides a way to load data from different sources, such as the local file system, Amazon S3 buckets, or Google Drives. The data is represented as a `RawDocument`, which provides a common abstraction for interacting with the data.

**Document Parsers**
Once data is loaded into `RawDocument`s, it must be parsed into a format usable by the other components of the library -- we call these `Document`s. The `DocumentParser` class defines how a `RawDocument` is converted to a `Document` for further use. A `Document` is composed of one or more smaller `DocumentFragment`s, which represent a smaller portion of the larger `Document`. A `DocumentFragment` should ideally represent some complete context of a portion of the larger `Document`: for example, a PDF's `Document` representation might have each of its pages represented by a `DocumentFragment`.

### Data Transformations
Although `Document`s may be used directly after they are parsed from a data source, it is often beneficial to first convert them into a format that will benefit the end use case of the data. This includes, but is not limited to: transforming unstructed data to structured data; creating vector embeddings from the data; summarizing or extracting additional information from the data. The `Transformer` class serves as a basis for implementing the transformations of `Document`s into these formats.

### Data Indexing
Transformed `Document`s and other data can be stored in various underlying indexes or data stores in order to be retrieved for future use in the LLM flow. One very common example is the indexing of contextual data as vector embeddings in a vector database, to be retrieved at LLM completion request time in order to augment the prompt context. The library supports a number of such data stores and indexes out-of-the-box, with useful abstractions to leverage for additional implementations.


## Data Retrieval
Data that is stored in an underyling data store or index can be obtained using a `Retriever` which has knowledge of the underlying implementation and its relevant query structure. Retrievers are not limited to retrieving data from a single underlying data store: they can support any desired custom retrieval logic for more complex use cases (composition of retrievers, heuristics such as staleness, etc.).


## Completion Generators
`CompletionGenerator`s define how an input query or prompt (e.g. from a user) is used to generate a completion response from an underlying `CompletionModel`, and what the final result looks like. Similar to retrievers, completion generators are not limited to performing completion requests to a model, but can support any desired custom logic between an input query and finalized response.

## Access Control
Proper access control is essential for leveraging data in semantic retrieval flows. The access control implementation in this framework differs slightly depending on the language used. A future iteration will consolidate both languages to a single implementation.

### Typescript
The typescript library leverages a concept of `ResourceAccessPolicy` to define which identities have access to requested resources. During ingestion, the `ResourceAccessPolicies` for source `RawDocument`s is specified in the associated `DocumentMetadataDB` entries. 
Post-ingestion, any access to an underlying data store is performed with the use of an `AccessPassport`, which is a mapping of all the resources an end-user has access to, to that user's `AccessIdentity` (e.g. authentication credentials) for that resource. When an attempt is made to access data belonging to a particular resource, the relevant `ResourceAccessPolicy` tests the user's `AccessIdentity` to determine their permission for that resource.

During retrieval, a `DocumentRetriever` will test whether the requestor has access to read the documents by calling the `testDocumentReadPermission` to filter out non-readable documents from those returned. For non-document resource accesses, the `testPolicyPermission` function on the associated `ResourceAccessPolicy` can be used to determine which permissions the identity has for the resource.


### Python
The python library leverages a `user_access_function` and `viewer_identity` as part of retrieval. The lowest point of data access calls the `user_access_function` to determine if the `viewer_identity` has permissions to access a particular resource: if a retriever retrieves data directly, the access check is performed in the retriever; otherwise, if the retriever queries data from an underlying data store, the access check is performed by the data store.


#### Changelog
Pypi v0.1.3

This version contains some simple new helper functions for constructing custom config objects from .env and CLI arguments. Also includes README updates applying to both Python and Typescript.

```
changeset:   2a64a87d1890059ceb1d5f21d30f4cb4da9d0bf6
user:        Jonathan Lessinger <jonathan@lastmileai.dev>
date:        Mon, 13 Nov 2023 14:10:09 -0500
summary:     [SR-PY] add config-related helper functions

changeset:   c565aa8aa8aa70cde160ca9b779558423b09f511
user:        Ryan Holinshead <>
date:        Sun, 12 Nov 2023 14:28:59 -0500
summary:     Minor Updates to README: Fill out Access Control section

changeset:   cd438aa90582627f706eb59122da62d5208ddca1
user:        Jonathan Lessinger <jonathan@lastmileai.dev>
date:        Mon, 13 Nov 2023 16:55:14 -0500
summary:     [EZ][SR-PY] one small type fix
```