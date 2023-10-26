from semantic_retrieval.prompts.prompt import IPrompt


class PromptTemplate(IPrompt):
    def __init__(self, template, params=None):
        self.template = template
        self.parameters = params if params is not None else {}

    def set_parameters(self, params):
        self.parameters = params

    def resolve_template(self, parameters=None):
        # TODO: Implement using handlebars. Merge parameters with self.parameters and have
        # method parameters take precedence.
        return ""

    def to_string(self):
        return self.resolve_template()
