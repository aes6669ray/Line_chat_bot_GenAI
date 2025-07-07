import asyncio
from googletrans import Translator

async def trans_text(text: str):
    translator = Translator()
    result = await translator.translate(text=text, dest='en')
    return result.text

# 包裝成同步調用
def translate(text: str):
    return asyncio.run(trans_text(text))

