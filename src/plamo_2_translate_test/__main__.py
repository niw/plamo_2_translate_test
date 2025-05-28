import asyncio

from .translator import Translation, Translator


async def main():
    translator = Translator()
    await translator.load_model_and_tokenizer()

    while True:
        english = input("English: ")

        print("Japanese: ", end="", flush=True)
        async for word in translator.stream(Translation(text=english)):
            print(word, end="", flush=True)


asyncio.run(main())
