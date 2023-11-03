import { JSONObject } from "./jsonTypes";

/**
 * Implement this interface to get property bags for metadata and free-form attributes.
 */
export interface Attributable {
  // Any JSON-serializable metadata (like configuration settings) associated with the object.
  metadata?: JSONObject;

  // A general property bag associated with this object.
  attributes?: JSONObject;
}

/**
 * Implement this interface to get a unique identifier for the object.
 */
export interface Identifiable {
  // The unique identifier for the object.
  id: string;
}
