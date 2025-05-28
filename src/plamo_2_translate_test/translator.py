import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import AsyncGenerator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.generation.streamers import AsyncTextIteratorStreamer


_MODEL_NAME = "pfnet/plamo-2-translate"
_STOP_STRING = "<|plamo:op|>"


@dataclass
class Translation:
    text: str
    from_language: str | None = None
    to_language: str | None = None

    @property
    def prompt(self) -> str:
        from_language = self.from_language or "English"
        to_language = self.to_language or "Japanese"

        return f"""<|plamo:op|>dataset
translation
<|plamo:op|>input lang={from_language}
{self.text}
<|plamo:op|>output lang={to_language}
"""

class Translator:
    def __init__(self, device: str | None = None):
        self._device = (
            device if device else "mps" if torch.mps.is_available() else "cpu"
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.__background_model = None
        self.__background_tokenizer = None

    @property
    def _background_model(self) -> PreTrainedModel:
        if not self.__background_model:
            self.__background_model = AutoModelForCausalLM.from_pretrained(
                _MODEL_NAME, trust_remote_code=True
            )
            self.__background_model.to(self._device)
        return self.__background_model

    @property
    def _background_tokenizer(self) -> PreTrainedTokenizer:
        if not self.__background_tokenizer:
            self.__background_tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME, trust_remote_code=True
            )
        return self.__background_tokenizer

    @property
    async def _tokenizer(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self._background_tokenizer
        )

    def _background_load_model_and_tokenizer(self):
        _ = self._background_model
        _ = self._background_tokenizer

    async def load_model_and_tokenizer(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor, self._background_load_model_and_tokenizer
        )

    def _background_generate(self, translation: Translation) -> str:
        self._background_load_model_and_tokenizer()

        prompt = translation.prompt
        inputs = self._background_tokenizer(prompt, return_tensors="pt")
        inputs.to(self._background_model.device)

        # TODO: Remove type: ignore when we have intersection type support for
        # type PreTrainedModel & GenerationMixin.
        outputs = self._background_model.generate(
            **inputs,
            tokenizer=self._background_tokenizer,
            max_length=1024,
            stop_strings=_STOP_STRING,
        )  # type: ignore

        decoded = self._background_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        output = decoded[0]
        output = output[len(prompt) : len(output) - len(_STOP_STRING)]

        return output.strip()

    async def __call__(self, text) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._background_generate, text)

    def _background_generate_stream(
        self, translation: Translation, *, streamer: AsyncTextIteratorStreamer
    ):
        inputs = self._background_tokenizer(translation.prompt, return_tensors="pt")
        inputs.to(self._background_model.device)

        # TODO: Remove type: ignore when we have intersection type support for
        # type PreTrainedModel & GenerationMixin.
        self._background_model.generate(
            **inputs,
            streamer=streamer,
            tokenizer=self._background_tokenizer,
            max_length=1024,
            stop_strings=_STOP_STRING,
        )  # type: ignore

    async def stream(self, translation: Translation) -> AsyncGenerator[str, None]:
        # NOTE: I think usage of `AutoTokenizer` is wrong, which is factory class.
        streamer = AsyncTextIteratorStreamer(
            await self._tokenizer,  # type: ignore
            skip_prompt=True,
            skip_special_tokens=True,
        )
        future = self._executor.submit(
            self._background_generate_stream, translation, streamer=streamer
        )

        async for token in streamer:
            if token == _STOP_STRING:
                break
            yield token

        future.result()
