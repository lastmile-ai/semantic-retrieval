import { compile } from "handlebars";
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

  resolveTemplate(parameters?: PromptTemplateParameters): string {
    const compiledTemplate = compile(this.template);
    return compiledTemplate(parameters ?? this.parameters);
  }

  async toString(): Promise<string> {
    return this.resolveTemplate();
  }
}
