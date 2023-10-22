import { CallbackManager, GetAccessIdentityEvent, RegisterAccessIdentityEvent, Traceable } from "../utils/callbacks";
import { AccessIdentity } from "./accessIdentity";

/**
 * Class for maintaining resource-to-access-identity mapping for a given identity (e.g. user).
 */
export class AccessPassport implements Traceable {
  accessIdentities: Map<string, AccessIdentity> = new Map();
  callbackManager?: CallbackManager;

  constructor() {}

  register(accessIdentity: AccessIdentity) {
    this.accessIdentities.set(accessIdentity.resource, accessIdentity);
    const event: RegisterAccessIdentityEvent = {name: 'onRegisterAccessIdentity', identity: accessIdentity};
    this.callbackManager?.runCallbacks(event);
  }

  getIdentity(resource: string) {
    const accessIdentity = this.accessIdentities.get(resource);

    const event: GetAccessIdentityEvent = {name: 'onGetAccessIdentity', resource, identity: accessIdentity};
    this.callbackManager?.runCallbacks(event);
    
    return accessIdentity;

  }
}
