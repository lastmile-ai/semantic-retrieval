/**
 * Implement this interface to get property bags for metadata and free-form attributes.
 */
export interface Attributable {
  // Any JSON-serializable metadata (like configuration settings) associated with the object.
  metadata: { [key: string]: string };

  // A general property bag associated with this object.
  attributes: { [key: string]: string };
}

/**
 * Implement this interface to get a unique identifier for the object.
 */
export interface Identifiable {
  // The unique identifier for the object.
  id: string;
}
