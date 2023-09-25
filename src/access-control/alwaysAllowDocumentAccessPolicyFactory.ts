import { RawDocument } from "../document/document";
import { DocumentAccessPolicyFactory } from "./documentAccessPolicyFactory";
import { AlwaysAllowAccessPolicy } from "./policies/alwaysAllowAccessPolicy";
import { ResourceAccessPolicy } from "./resourceAccessPolicy";

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
