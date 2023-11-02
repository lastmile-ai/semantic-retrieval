import asyncio
import sys
from aiconfig.Config import AIConfigRuntime
import os

import dotenv

DIR = "python/src/semantic_retrieval/evaluation/aiconfig"

runtime = AIConfigRuntime.from_config(os.path.join(DIR, "parameterized_data_config.json"))


async def main():
    dotenv.load_dotenv()
    print("Running")
    try:
        result = await runtime.run(
        "prompt1",
        {
            "sql_language": "hiveql",
        })
        print(f"{result=}")
        return 0
    except Exception as e:
        print(f"{e=}")
        return -1

if __name__ == "__main__":    
    res = asyncio.run(main())
    sys.exit(res)
        