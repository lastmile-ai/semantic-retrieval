# semantic-retrieval

This project is under development.
**Please refer to the [Semantic Retrieval doc](https://docs.google.com/document/d/1XO4lj-cpFgd6Gl4VkkDsz7K5Y03quW3G1qXOyVaKyY8/edit?pli=1#heading=h.pml5g74m7m5n) for more details**

# Development
This library uses yarn for package management.
```
git clone https://github.com/lastmile-ai/semantic-retrieval
cd semantic-retrieval
yarn
```

# Testing
To run all tests, do:
```
yarn test
```

To run an individual test, use `yarn test <filename.test.ts>`.

# Examples & Demos
Example scripts, such as `localFileIngestion` can be run via `ts-node`:
```
npx ts-node examples/ingestion/localFileIngestion.ts
```
Ensure you have the correct openai and pinecone environment variables set up in a `.env` file if not
passing the values through the constructors in the scripts:
```
OPENAI_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_API_KEY=
```
if using `PineconeVectorDB`, ensure you have an Pinecone index created beforehand and set the proper `indexName`.

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
Transformed `Document`s and other data can be stored in various underlying indexes or data stores in order to be retrieved for future use in the LLM flow. One very common example is the indexing of contextual data as vector embeddings in a vector database, to be retrieved at LLM completion reqeust time in order to augment the prompt context. The library supports a number of such data stores and indexes out-of-the-box, with useful abstractions to leverage for additional implementations.


## Data Retrieval
Data that is stored in an underyling data store or index can be obtained using a `Retriever` which has knowledge of the underlying implementation and its relevant query structure. Retrievers are not limited to retrieving data from a single underlying data store: they can support any desired custom retrieval logic for more complex use cases (composition of retrievers, heuristics such as staleness, etc.).


## Completion Generators
`CompletionGenerator`s define how an input query or prompt (e.g. from a user) is used to generate a completion response from an underlying `CompletionModel`, and what the final result looks like. Similar to retrievers, completion generators are not limited to performing completion requests to a model, but can support any desired custom logic between an input query and finalized response.

## Access Control
// TODO