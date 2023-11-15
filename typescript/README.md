# Development
This library uses yarn for package management.
```
git clone https://github.com/lastmile-ai/semantic-retrieval
cd semantic-retrieval/typescript
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


# Quick dev (using local SR package in other local projects)

Developing and testing the semantic-retrieval package can be done locally using another local project containing semantic-retrieval as a dependency, following the steps below:

### In semantic-retrieval Project

1. Clone (https://github.com/lastmile-ai/lastmile-att-demo.git) (or pull) and cd to typescript directory
2. Create a `.env` file with your API keys, like this:
```
OPENAI_API_KEY="my-openai-key"
PINECONE_API_KEY="my-pinecone-key"
PINECONE_ENVIRONMENT="my-pinecone-environment"
```
3. In local SR repo, cd to typescript directory. Build and pack a local package:
```
yarn build && yarn pack
```

NOTE: If a local tarball already exists, remove it before running build and pack to prevent caching issues
```
rm -rf ./dist && rm semantic-retrieval-v0.0.1.tgz && yarn build && yarn pack
```

### In Other Local Projects

4. Update `semantic-retrieval` dependency in `package.json` in other local project to point to the local package, e.g:
```
"semantic-retrieval": "file:/Users/ryanholinshead/Projects/semantic-retrieval/typescript/semantic-retrieval-v0.0.1.tgz",
```
5. Run `yarn` to install the local package

NOTE: If the local package was previously installed, clear out node modules and yarn cache before installing to prevent caching issues
```
rm -rf node_modules && rm yarn.lock && yarn cache clean && yarn 
```