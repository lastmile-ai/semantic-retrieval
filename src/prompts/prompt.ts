/**
 * Prompts are string-serializable objects that can be used to generate completions from an LLM
 */
export interface IPrompt {
  toString(): Promise<string>;
}

export class Prompt implements IPrompt {
  prompt: string;

  constructor(prompt: string) {
    this.prompt = prompt;
  }

  async toString() {
    return this.prompt;
  }
}
