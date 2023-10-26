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
        accessPassport = params.get("accessPassport")
        prompt = params["prompt"]
        retriever = params["retriever"]
        modelParams = {
            k: v for k, v in params.items() if k not in ["accessPassport", "prompt", "retriever"]
        }

        queryPrompt = prompt if isinstance(prompt, str) else prompt.to_string()

        contextDocs = await retriever.retrieve_data(
            {"accessPassport": accessPassport, "query": await self.get_retrieval_query(params)}
        )

        contextChunksPromises = []
        for doc in contextDocs:
            for fragment in doc["fragments"]:
                contextChunksPromises.append(fragment.get_content())

        context = "\n".join(await asyncio.gather(*contextChunksPromises))
        ragPromptTemplate = params.get("ragPromptTemplate") or PromptTemplate(DEFAULT_RAG_TEMPLATE)
        ragPromptTemplate.set_parameters({"prompt": queryPrompt, "context": context})

        response = await self.model.run({**modelParams, "prompt": ragPromptTemplate})

        if self.callback_manager:
            await self.callback_manager.run_callbacks(
                {"name": "onRunCompletionGeneration", "params": params, "response": response}
            )

        return response
