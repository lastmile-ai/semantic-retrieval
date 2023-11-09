import { Attributable } from "../common/base";
import { GLOBAL_RESOURCE } from "./resourceAccessPolicy";

/**
 * Represents a single identity (e.g. auth credentials) for accessing a resource (e.g. Google Drive)
 */
export interface AccessIdentity extends Attributable {
  resource: string;
}

/**
 * An identity that is registered for all globally-scoped resources in all AccessPassports
 * This allows for globally-scoped access policies (e.g. AlwaysAllowAccessPolicy) to leverage
 * the identity-based resource access policy checks.
 */
export const GLOBAL_RESOURCE_IDENTITY: AccessIdentity = {
  resource: GLOBAL_RESOURCE,
};
