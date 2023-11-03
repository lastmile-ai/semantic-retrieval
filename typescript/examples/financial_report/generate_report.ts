#!/usr/bin/env ts-node
// Description: Script to generate a report for financial advisor clients. Run this after ingest_data.ts
// Usage Example: npx ts-node examples/typescript/financial_report/generate_report.ts -r advisor -c sarmad -p my-index -n my-namespace

import { OptionValues, program } from "commander";
import { AccessPassport } from "../../../typescript/src/access-control/accessPassport";
import { PineconeVectorDB } from "../../../typescript/src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../../typescript/src/document/metadata/inMemoryDocumentMetadataDB";
import { CSVRetriever } from "../../../typescript/src/retrieval/csvRetriever";
import { VectorDBDocumentRetriever } from "../../../typescript/src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { OpenAIEmbeddings } from "../../../typescript/src/transformation/embeddings/openAIEmbeddings";
import { AdvisorIdentity } from "./components/access_control/advisorIdentity";
import { AdminIdentity } from "./components/access_control/adminIdentity";
import {
  CompanyProfiles,
  FinancialReportDocumentRetriever,
  PortfolioData,
} from "./components/financialReportDocumentRetriever";
import dotenv from "dotenv";
import { ResourceAccessPolicy } from "../../../typescript/src/access-control/resourceAccessPolicy";
import { AlwaysAllowAccessPolicy } from "../../../typescript/src/access-control/policies/alwaysAllowAccessPolicy";
import fs from "fs/promises";
import { FinancialReportGenerator } from "./components/financialReportGenerator";
import { SecretReportAccessPolicy } from "./components/access_control/secretReportAccessPolicy";
import { AccessIdentity } from "../../../typescript/src/access-control/accessIdentity";
import {
  CallbackManager,
  RetrieveDataEvent,
  RetrievedFragmentPolicyCheckFailedEvent,
  RunCompletionGenerationEvent,
  RunCompletionRequestEvent,
  RunCompletionResponseEvent,
} from "../../../typescript/src/utils/callbacks";
import { v4 as uuid } from "uuid";

dotenv.config();

program
  .name("generate_report")
  .description(
    "Script to generate a report for financial advisor clients. Run this after ingest_data.ts"
  );

program.option(
  "-r, --role [ROLE]",
  "specify the user role to generate a report for (admin or advisor)",
  "advisor"
);

program.option(
  "-c, --client_id [CLIENT_ID]",
  "specify the client id to generate a report for. One of sarmad or tanya",
  "sarmad"
);

program.option(
  "-p, --pinecone_index [PINECONE_INDEX]",
  // Make sure this matches your Pinecone index name & it has 1536 dimensions for openai embeddings
  "specify the name of the pinecone index to ingest the documents into",
  "test-financial-report" // default pinecone index name
);

program.option(
  "-n, --pinecone_namespace [PINECONE_NAMESPACE]",
  // Make sure this matches the namespace created from ingest_data script
  "specify the namespace in the pinecone index containing document embeddings",
  "ea4bcf44-e0f3-46ff-bf66-5b1f9e7502df" // default pinecone namespace from 'good' ingest_data run
);

program.option(
  "-t, --topic [TOPIC]",
  "specify the topic to generate a report for",
  "Recovery from the COVID-19 pandemic"
);

program.option("-v, --verbose", "specify whether to print verbose logs", false);

program.parse(process.argv);

async function main() {
  const options = program.opts();
  const {
    indexName,
    namespace,
    clientId,
    accessIdentity,
    topic,
    verboseLogging,
  } = getOptions(options);

  // Load the metadataDB persisted from ingest_data script
  const metadataDB = await InMemoryDocumentMetadataDB.fromJSONFile(
    "examples/financial_report/metadataDB.json",
    (key, value) => {
      if (key === "accessPolicies") {
        // deserialize access policies to their instances
        return (value as ResourceAccessPolicy[]).map(
          (policy: ResourceAccessPolicy) => {
            switch (policy.policy) {
              case "AlwaysAllowAccessPolicy":
                return new AlwaysAllowAccessPolicy();
              case "SecretReportAccessPolicy":
                return new SecretReportAccessPolicy();
              default:
                return policy;
            }
          }
        );
      }
      return value;
    }
  );

  const callbackManager = getLoggingCallbackManager(verboseLogging);

  const vectorDB = await new PineconeVectorDB({
    indexName,
    namespace,
    embeddings: new OpenAIEmbeddings(),
    metadataDB,
  });

  const documentRetriever = new VectorDBDocumentRetriever({
    vectorDB,
    metadataDB,
    callbackManager,
  });

  const portfolioRetriever = new CSVRetriever<PortfolioData>(
    `../examples/example_data/financial_report/portfolios/${clientId}_portfolio.csv`
  );

  const companyProfilesRetriever = new CSVRetriever<CompanyProfiles>(
    "../examples/example_data/financial_report/company_profiles.csv"
  );

  const accessPassport = new AccessPassport();
  accessPassport.register(accessIdentity);

  const retriever = new FinancialReportDocumentRetriever({
    documentRetriever,
    portfolioRetriever,
    companyProfilesRetriever,
    metadataDB,
    callbackManager,
  });

  const generator = new FinancialReportGenerator(callbackManager);

  console.log("Generating report...");
  const report = await generator.run({
    prompt: topic,
    accessPassport,
    retriever,
  });

  console.log("Writing report to disk...");
  await fs.writeFile("examples/typescript/financial_report/report.txt", report);
  console.log(
    "Report written to examples/typescript/financial_report/report.txt"
  );
}

function getLoggingCallbackManager(verboseLogging: boolean) {
  return new CallbackManager(`generate-report-${uuid()}`, {
    onRetrieveData: [
      async (event: RetrieveDataEvent) => {
        if (verboseLogging) {
          console.log("Retrieved data: ", event.data);
        } else {
          console.log("Retrieved data");
        }
      },
    ],
    onRunCompletionGeneration: [
      async (event: RunCompletionGenerationEvent<any>) => {
        if (verboseLogging) {
          console.log("Generated completion: ", event.response);
        } else {
          console.log("Generated completion");
        }
      },
    ],
    onRunCompletionRequest: [
      async (event: RunCompletionRequestEvent) => {
        if (verboseLogging) {
          console.log("Performing completion request: ", event.params);
        } else {
          console.log("Performing completion request");
        }
      },
    ],
    onRunCompletionResponse: [
      async (event: RunCompletionResponseEvent) => {
        if (verboseLogging) {
          console.log("Received completion response: ", event.response);
        } else {
          console.log("Received completion response");
        }
      },
    ],
    onRetrievedFragmentPolicyCheckFailed: [
      async (event: RetrievedFragmentPolicyCheckFailedEvent) => {
        console.log("Fragment policy check failed: ", {
          fragmentId: event.fragment.fragmentId,
          documentId: event.fragment.documentId,
          policy: event.policy.policy,
        });
      },
    ],
  });
}

function getOptions(options: OptionValues) {
  const {
    pinecone_index: indexName,
    pinecone_namespace: namespace,
    client_id: clientId,
    role,
    topic,
    verbose,
  } = options;

  if (typeof indexName !== "string") {
    throw new Error("no index name or default specified");
  }

  if (typeof namespace !== "string") {
    throw new Error("no namespace or default specified");
  }

  if (typeof clientId !== "string") {
    throw new Error("no client id or default specified");
  }

  if (typeof topic !== "string") {
    throw new Error("no topic or default specified");
  }

  if (clientId !== "sarmad" && clientId !== "tanya") {
    throw new Error(
      "invalid client id specified. Must be one of sarmad or tanya"
    );
  }

  if (typeof role !== "string") {
    throw new Error("no role or default specified");
  }

  let accessIdentity: AccessIdentity;

  switch (role) {
    case "admin":
      accessIdentity = new AdminIdentity();
      break;
    case "advisor":
      accessIdentity = new AdvisorIdentity();
      break;
    default:
      throw new Error(
        "invalid role specified. Must be one of admin or advisor"
      );
  }

  return {
    indexName,
    namespace,
    clientId,
    accessIdentity,
    topic,
    verboseLogging: verbose !== false,
  };
}

main()
  .then(() => {
    console.log("Done!");
  })
  .catch((err: any) => {
    console.error("Error:", err);
    process.exit(1);
  })
  .finally(() => {
    process.exit(0);
  });

export {};
