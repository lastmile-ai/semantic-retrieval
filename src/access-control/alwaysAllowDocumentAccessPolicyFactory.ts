import { RawDocument } from "../document/document.js";
import { DocumentAccessPolicyFactory } from "./documentAccessPolicyFactory.js";
import { AlwaysAllowAccessPolicy } from "./policies/alwaysAllowAccessPolicy.js";
import { ResourceAccessPolicy } from "./resourceAccessPolicy.js";

export class AlwaysAllowDocumentAccessPolicyFactory
  implements DocumentAccessPolicyFactory
{
  constructor() {}

  async getAccessPolicies(
    _rawDocument: RawDocument,
  ): Promise<ResourceAccessPolicy[]> {
    return [new AlwaysAllowAccessPolicy()];
  }
}
