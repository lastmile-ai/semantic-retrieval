import { AccessIdentity } from "./accessIdentity";

/**
 * Class for maintaining resource-to-access-identity mapping for a given identity (e.g. user).
 */
export class AccessPassport {
  accessIdentities: Map<string, AccessIdentity> = new Map();

  constructor() {}

  register(accessIdentity: AccessIdentity) {
    this.accessIdentities.set(accessIdentity.resource, accessIdentity);
  }

  getIdentity(resource: string) {
    return this.accessIdentities.get(resource);
  }
}
