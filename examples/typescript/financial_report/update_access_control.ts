#!/usr/bin/env ts-node
// Description: Script to update access controls in the metadataDB for an ingested financial report document
// Usage Example: npx ts-node examples/typescript/financial_report/update_access_control.ts -c AAPL -a AlwaysAllowAccessPolicy

import { program } from "commander";
import { AlwaysAllowAccessPolicy } from "../../../src/access-control/policies/alwaysAllowAccessPolicy";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import { SecretReportAccessPolicy } from "./components/access_control/secretReportAccessPolicy";

program
  .name("update_access_control")
  .description(
    "Script to update access controls in the metadataDB for an ingested financial report document"
  );

program.option(
  "-c, --company <COMPANY>",
  "specify which company 10k to apply the access control to"
);

program.option(
  "-a, --access <ACCESS>",
  "specify which access control to apply to the company. One of AlwaysAllowAccessPolicy or SecretReportAccessPolicy"
);

program.parse(process.argv);

async function main() {
  console.log("Applying access controls to company report...");

  const options = program.opts();

  const company = options.company;
  if (typeof company !== "string") {
    throw new Error("company must be specified");
  }

  const access = options.access;
  if (typeof access !== "string") {
    throw new Error("access must be specified");
  }

  const accessPolicy = getAccessPolicy(access);

  // Load the metadataDB persisted from ingest_data script
  const metadataDB = await InMemoryDocumentMetadataDB.fromJSONFile(
    "examples/typescript/financial_report/metadataDB.json"
  );

  const relevantDocIds = await metadataDB.queryDocumentIds({
    type: "string_field",
    fieldName: "name",
    fieldValue: company,
    matchType: "includes",
  });

  if (relevantDocIds.length === 0) {
    throw new Error(`No documents found for company: ${company}`);
  }

  await Promise.all(
    relevantDocIds.map(async (docId) => {
      const metadata = await metadataDB.getMetadata(docId);
      metadata.accessPolicies = [accessPolicy];
      await metadataDB.setMetadata(docId, metadata);
    })
  );

  // Persist metadataDB to disk for loading in the other scripts
  await metadataDB.persist(
    "examples/typescript/financial_report/metadataDB.json"
  );

  console.log(
    "Successfully updated access control for company report documents"
  );
}

function getAccessPolicy(access: string) {
  switch (access) {
    case "AlwaysAllowAccessPolicy":
      return new AlwaysAllowAccessPolicy();
    case "SecretReportAccessPolicy":
      return new SecretReportAccessPolicy();
    default:
      throw new Error(`Unknown access policy: ${access}`);
  }
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
