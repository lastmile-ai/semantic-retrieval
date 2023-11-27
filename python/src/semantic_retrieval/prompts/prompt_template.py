from typing import Dict, Optional

from semantic_retrieval.prompts.prompt import IPrompt

PromptTemplateParameters = Dict[str, str]


class PromptTemplate(IPrompt):
    def __init__(self, template: str, params: PromptTemplateParameters):
        self.template = template
        self.parameters = params

    def set_parameters(self, params: PromptTemplateParameters):
        self.parameters = params

    def resolve_template(
        self, parameters: Optional[PromptTemplateParameters] = None
    ):
        return ""

    async def to_string(self) -> str:
        return self.resolve_template()
