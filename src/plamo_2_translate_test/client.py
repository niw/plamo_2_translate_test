from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import AsyncGenerator

import httpx

from .translator import Translation


class Client:
    def __init__(self, url: str):
        self._client = httpx.AsyncClient(timeout=600)
        self._url = url

    @property
    def _endpoint_url(self) -> str:
        return f"{self._url}/translate"

    async def stream(self, translation: Translation) -> AsyncGenerator[str, None]:
        request = asdict(translation)
        request["stream"] = True

        async with self._client.stream(
            "POST", self._endpoint_url, json=request
        ) as response:
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]

                elif line.startswith("data: "):
                    data = line[6:]
                    match event:
                        case "delta":
                            yield data
                        case "done":
                            break

                else:
                    event = None

    async def __call__(self, translation: Translation) -> str:
        response = await self._client.post(self._endpoint_url, json=asdict(translation))
        response.raise_for_status()
        result = response.json()
        return result.get("result")
