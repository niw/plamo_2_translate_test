from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from .translator import Translation, Translator


__translator: Translator | None = None


def _translator() -> Translator:
    global __translator
    if __translator is None:
        __translator = Translator()
    return __translator


@asynccontextmanager
async def _lifespan(app: FastAPI):
    translator = _translator()
    await translator.load_model_and_tokenizer()
    yield


app = FastAPI(lifespan=_lifespan)


@dataclass
class _TranslationRequest(Translation):
    stream: bool = False


@dataclass
class _TranslationResponse:
    result: str


@app.post("/translate")
async def _translate(
    request: _TranslationRequest,
    translator: Annotated[Translator, Depends(_translator)],
):
    if request.stream:
        generator = translator.stream(request)

        async def event_stream() -> AsyncGenerator[str, None]:
            async for chunk in generator:
                yield f"event: delta\ndata: {chunk}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        result = await translator(request)
        return _TranslationResponse(result=result)
