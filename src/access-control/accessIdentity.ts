import { Attributable } from "../common/base";

/**
 * Represents a single identity (e.g. auth credentials) for accessing a resource (e.g. Google Drive)
 */
export interface AccessIdentity extends Attributable{
    resource: string;
}