from semantic_retrieval.generator.retrieval_augmented_generation.rag_completion_generator import (
    RAGCompletionGenerator,
)


class VectorDBRAGCompletionGeneratorParams:
    def __init__(self):
        self.retriever = VectorDBDocumentRetriever()
        self.retrievalQuery = None


class VectorDBRAGCompletionGenerator(RAGCompletionGenerator):
    async def get_retrieval_query(self, params):
        prompt = params["prompt"]
        retrieval_query = params.get("retrievalQuery", None)

        if isinstance(prompt, str):
            text = prompt
        else:
            text = prompt.to_string()

        top_k = retrieval_query["topK"] if retrieval_query else 3

        query = {
            "topK": top_k,
            "text": text,
        }

        if self.callback_manager:
            await self.callback_manager.run_callbacks(
                {
                    "name": "onGetRAGCompletionRetrievalQuery",
                    "params": params,
                    "query": query,
                }
            )

        return query
