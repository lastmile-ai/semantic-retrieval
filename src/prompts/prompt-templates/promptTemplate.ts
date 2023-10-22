import { IPrompt } from "../prompt";

export type PromptTemplateParameters = { [key: string]: string };
/**
 * A class for defining reusable prompts with template strings. This base class
 * uses handlebars for template resolution by default.
 */
export class PromptTemplate implements IPrompt {
  template: string;
  parameters: PromptTemplateParameters = {};

  constructor(template: string, params?: PromptTemplateParameters) {
    this.template = template;
    this.parameters = params ?? this.parameters;
  }

  setParameters(params: PromptTemplateParameters) {
    this.parameters = params;
  }

  resolveTemplate(_parameters?: PromptTemplateParameters): string {
    // TODO: Implement using handlebars. Merge parameters with this.parameters and have
    // method parameters take precedence.
    return "";
  }

  async toString(): Promise<string> {
    return this.resolveTemplate();
  }
}
