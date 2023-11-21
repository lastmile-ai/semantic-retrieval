from typing import Any, Dict, List

from aiconfig import AIConfigRuntime, ExecuteResult
from aiconfig.model_parser import InferenceOptions


async def generate(ai_config_path: str, params: Dict[str, str]) -> str:
    runtime = AIConfigRuntime.from_config(ai_config_path)

    def callback(data: Any, acc: Any, output_num: int) -> None:
        if "content" in data:
            content = data["content"]
            if content:
                print(content, end="", flush=True)

    options = InferenceOptions(stream_callback=callback)

    out_list: List[ExecuteResult] = await runtime.run(
        "rag_complete", params=params, options=options
    )  # type: ignore [LAS-441]
    assert len(out_list) == 1
    out: ExecuteResult = out_list[0]
    text = runtime.get_output_text("rag_complete", output=out)  # type: ignore [LAS-441]

    return text


async def resolve_ai_config(
    ai_config_path: str, params: Dict[str, str]
) -> Any:
    runtime = AIConfigRuntime.from_config(ai_config_path)
    return await runtime.resolve("rag_complete", params=params)


def ai_config_metadata_lookup(ai_config_path: str, key: str) -> Dict[str, str]:
    runtime = AIConfigRuntime.from_config(ai_config_path)
    md = getattr(runtime.metadata, key)
    return md
