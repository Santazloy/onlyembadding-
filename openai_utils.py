# openai_utils.py

import os
import asyncio
import re
import openai

# Попытка создать клиент через новый класс OpenAI
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
except ImportError:
    # fallback на модуль openai
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    _client = openai


async def get_embedding(text: str) -> list[float]:
    """
    Универсальный эмбеддинг (text-embedding-3-large).
    """
    loop = asyncio.get_event_loop()

    def _call():
        if hasattr(_client, "embeddings"):
            return _client.embeddings.create(model="text-embedding-3-large", input=text)
        else:
            return _client.Embedding.create(model="text-embedding-3-large", input=text)

    try:
        resp = await loop.run_in_executor(None, _call)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[OpenAI] get_embedding error: {e}")
        return []


async def transcribe_audio(file_path: str) -> str:
    """
    Универсальная транскрипция через Whisper.
    """
    loop = asyncio.get_event_loop()

    def _call():
        if hasattr(_client, "audio"):
            return _client.audio.transcriptions.create(model="whisper-1", file=open(file_path, "rb"))
        else:
            return _client.Audio.transcribe("whisper-1", open(file_path, "rb"))

    try:
        resp = await loop.run_in_executor(None, _call)
        return resp["text"]
    except Exception as e:
        print(f"[OpenAI] transcribe_audio error: {e}")
        return ""


async def generate_text(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 2000
) -> str:
    """
    Универсальная генерация текста через ChatCompletion.
    """
    loop = asyncio.get_event_loop()

    def _call():
        # Если модель для web-search-preview, убираем параметры temperature/top_p
        if "search-preview" in model:
            if hasattr(_client, "chat"):
                return _client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                return _client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
        # Иначе — полный набор параметров
        if hasattr(_client, "chat"):
            return _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0
            )
        else:
            return _client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0
            )

    try:
        resp = await loop.run_in_executor(None, _call)
        if hasattr(_client, "chat"):
            return resp.choices[0].message.content.strip()
        else:
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[OpenAI] generate_text error: {e}")
        return "Ошибка при генерации ответа."


async def generate_analysis_text(
    prompt: str,
    model: str = "gpt-4o"
) -> str:
    """
    Генерация развёрнутого анализа (например, для daily_report).
    """
    loop = asyncio.get_event_loop()

    def _call():
        if hasattr(_client, "chat"):
            return _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
        else:
            return _client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )

    try:
        resp = await loop.run_in_executor(None, _call)
        if hasattr(_client, "chat"):
            return resp.choices[0].message.content.strip()
        else:
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[OpenAI] generate_analysis_text error: {e}")
        return ""


def markdown_links_to_html(text: str) -> str:
    """
    [текст](url) -> <a href="url">текст</a>
    """
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    return re.sub(pattern, r'<a href="\2">\1</a>', text)
