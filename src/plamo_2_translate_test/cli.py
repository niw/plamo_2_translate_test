import argparse
import asyncio
import sys

import uvicorn

from .client import Client
from .translator import Translation, Translator
from .server import app


async def _server(host: str, port: int):
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def _translate(
    translation: Translation, server_url: str | None = None, stream: bool = False
):
    if server_url:
        client = Client(url=server_url)
        if stream:
            async for chunk in client.stream(translation):
                print(chunk, end="", flush=True)
            print()

        else:
            result = await client(translation)
            print(result)

    else:
        translator = Translator()
        if stream:
            async for result in translator.stream(translation):
                print(result)
        else:
            result = await translator(translation)
            print(result)


async def run():
    argparser = argparse.ArgumentParser(prog=__package__)
    subparsers = argparser.add_subparsers(dest="command", required=True)

    server_parser = subparsers.add_parser("server", help="Run translation server.")
    server_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the translation server on (default: '127.0.0.1').",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the translation server on (default: 8000).",
    )

    translate_parser = subparsers.add_parser(
        "translate", help="Translate text and quit."
    )
    translate_parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default="-",
        help="Text to translate. '-' to read from stdin (default: '-').",
    )
    translate_parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language (default: None).",
    )
    translate_parser.add_argument(
        "--server",
        action="store_true",
        default=False,
        help="Use translation server (default: False).",
    )
    translate_parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Translation server URL (default: 'http://127.0.0.1:8000').",
    )
    translate_parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream translation output (default: False).",
    )

    args = argparser.parse_args()

    match args.command:
        case "server":
            await _server(host=args.host, port=args.port)

        case "translate":
            if args.text == "-":
                loop = asyncio.get_running_loop()
                args.text = await loop.run_in_executor(None, sys.stdin.read)

            translation = Translation(text=args.text, language=args.lang)
            server_url = args.server_url if args.server else None
            await _translate(
                translation=translation, server_url=server_url, stream=args.stream
            )


def main():
    asyncio.run(run())
