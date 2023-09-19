import { Attributable } from "../common/base";

/**
 * Represents a single identity for accessing a resource (e.g. auth credentials)
 */
export interface AccessIdentity extends Attributable{
    resource: string;
}