import { AccessPassport } from "../access-control/accessPassport.js";
import { JSONValue } from "../common/jsonTypes.js";
import { BaseRetriever } from "../retrieval/retriever.js";

export type GeneratorParams<D> = {
  prompt: JSONValue;
  accessPassport?: AccessPassport;
  retriever?: BaseRetriever<D>;
};

/**
 * Generators are used for generating some response based on a prompt and
 * retrieved data (if applicable).
 */
export abstract class BaseGenerator<D, G> {
  protected abstract run(params: GeneratorParams<D>): Promise<G>;
}
