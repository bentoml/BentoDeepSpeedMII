import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

# import nest_asyncio
# nest_asyncio.apply()


MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

@bentoml.service(
    name="deepspeed-llama3.1-8b-instruct-service",
    traffic={
        "timeout": 300,
        "concurrency": 256, # Matches the default max_num_seqs in the VLLM engine
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class DeepSpeed:

    def __init__(self) -> None:
        import mii
        from transformers import AutoTokenizer

        import asyncio
        self.event_loop = asyncio.get_event_loop()
        self.server = mii.serve(MODEL_ID)
        self.client = mii.client(MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)

        # stream still WIP
        stream = self.client._request_async_response_stream([prompt], max_length=max_tokens)
        async for resp in stream:
            yield resp[0].generated_text


    @bentoml.on_shutdown
    def shutdown(self):
        print("shutting down!")
        self.client.terminate_server()
        print("shutdown finished!")
