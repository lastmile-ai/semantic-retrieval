export type JSONValue = string | number | boolean | JSONObject | JSONArray;

export interface JSONObject {
  [x: string]: JSONValue | undefined;
}

export interface JSONArray extends Array<JSONValue> {}
