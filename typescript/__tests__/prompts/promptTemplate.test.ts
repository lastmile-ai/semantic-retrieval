import { PromptTemplate } from "../../src/prompts/prompt-templates/promptTemplate";

describe("promptTemplate resolution", () => {
  test("should resolve a template with no parameters", () => {
    const template = new PromptTemplate("Hello, world!");
    expect(template.resolveTemplate()).toBe("Hello, world!");
  });

  test("should resolve a template with correct parameters passed in at resolve time", () => {
    const template = new PromptTemplate(
      "Hello, {{name}}! Here's some extra context: {{context}}"
    );
    expect(
      template.resolveTemplate({ name: "world", context: "this is a test" })
    ).toBe("Hello, world! Here's some extra context: this is a test");
  });

  test("should resolve a template with correct parameters set before resolution", () => {
    const template = new PromptTemplate(
      "Hello, {{name}}! Here's some extra context: {{context}}"
    );
    template.setParameters({ name: "world", context: "this is a test" });
    expect(template.resolveTemplate()).toBe(
      "Hello, world! Here's some extra context: this is a test"
    );
  });

  test("parameters set at resolution time take precedence", () => {
    const template = new PromptTemplate(
      "Hello, {{name}}! Here's some extra context: {{context}}"
    );
    template.setParameters({ name: "world", context: "this is a test" });
    expect(template.resolveTemplate({ name: "foo", context: "bar" })).toBe(
      "Hello, foo! Here's some extra context: bar"
    );
  });

  test("ignores parameters that are not used in the template", () => {
    const template = new PromptTemplate("Hello, {{name}}!");
    expect(
      template.resolveTemplate({ name: "world", context: "this is a test" })
    ).toBe("Hello, world!");
  });
});
