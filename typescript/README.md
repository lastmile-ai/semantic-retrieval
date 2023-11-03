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
Ensure you have the correct openai and pinecone environment variables set up in a `.env` file in the typescript directory
if not passing the values through the constructors in the scripts:
```
OPENAI_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_API_KEY=
```
if using `PineconeVectorDB`, ensure you have an Pinecone index created beforehand and set the proper `indexName`.