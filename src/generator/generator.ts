import { AccessPassport } from "../access-control/accessPassport";
import { JSONValue } from "../common/jsonTypes";
import { BaseRetriever } from "../retrieval/retriever";

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
