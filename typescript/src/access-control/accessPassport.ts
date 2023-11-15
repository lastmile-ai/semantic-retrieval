import {
  CallbackManager,
  GetAccessIdentityEvent,
  RegisterAccessIdentityEvent,
  Traceable,
} from "../utils/callbacks";
import { AccessIdentity, GLOBAL_RESOURCE_IDENTITY } from "./accessIdentity";
import { GLOBAL_RESOURCE } from "./resourceAccessPolicy";

export interface AccessPassportConfig {
  accessIdentities?: Map<string, AccessIdentity>;
  callbackManager?: CallbackManager;
}

/**
 * Class for maintaining resource-to-access-identity mapping for a given identity (e.g. user).
 */
export class AccessPassport implements Traceable {
  accessIdentities: Map<string, AccessIdentity> = new Map([
    [GLOBAL_RESOURCE, GLOBAL_RESOURCE_IDENTITY],
  ]);
  callbackManager?: CallbackManager;

  constructor(config?: AccessPassportConfig) {
    this.callbackManager = config?.callbackManager;
    if (config?.accessIdentities) {
      this.accessIdentities = config.accessIdentities;
      if (!this.accessIdentities.has(GLOBAL_RESOURCE)) {
        this.accessIdentities.set(GLOBAL_RESOURCE, GLOBAL_RESOURCE_IDENTITY);
      }
    }
  }

  register(accessIdentity: AccessIdentity) {
    this.accessIdentities.set(accessIdentity.resource, accessIdentity);
    const event: RegisterAccessIdentityEvent = {
      name: "onRegisterAccessIdentity",
      identity: accessIdentity,
    };
    this.callbackManager?.runCallbacks(event);
  }

  getIdentity(resource: string) {
    const accessIdentity = this.accessIdentities.get(resource);

    const event: GetAccessIdentityEvent = {
      name: "onGetAccessIdentity",
      resource,
      identity: accessIdentity,
    };
    this.callbackManager?.runCallbacks(event);

    return accessIdentity;
  }
}
