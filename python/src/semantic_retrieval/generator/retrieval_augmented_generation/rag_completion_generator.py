import asyncio

from semantic_retrieval.generator.completion_generator import (
    LLMCompletionGeneratorParams,
    LLMCompletionGenerator,
)

from semantic_retrieval.prompts.prompt_template import PromptTemplate


class RAGCompletionGeneratorParams(LLMCompletionGeneratorParams):
    def __init__(self, retriever, accessPassport=None, ragPromptTemplate=None):
        self.retriever = retriever
        self.accessPassport = accessPassport
        self.ragPromptTemplate = ragPromptTemplate


DEFAULT_RAG_TEMPLATE = "Answer the question based on the context below.\n\nContext:\n\n{{context}}\n\nQuestion: {{prompt}}\n\nAnswer:"


class RAGCompletionGenerator(LLMCompletionGenerator):
    async def get_retrieval_query(self, params):
        # Implement this method to construct the query for the underlying retriever
        pass

    async def run(self, params):
        # TODO
        pass
