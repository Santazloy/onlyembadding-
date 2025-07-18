# openai_utils.py
import openai
import asyncio

from config import OPENAI_API_KEY

# Инициализируем ключ
openai.api_key = OPENAI_API_KEY


async def get_embedding(text: str) -> list[float]:
    """
    Возвращает векторное представление текста с помощью text-embedding-3-large.
    """
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
        )
        # Теперь resp — объект CreateEmbeddingResponse, у него есть атрибут .data
        return resp.data[0].embedding
    except Exception as e:
        print(f"[OpenAI] Ошибка при получении эмбеддинга: {e}")
        return []


async def transcribe_audio(file_path: str) -> str:
    """
    Транскрибирует аудио через Whisper.
    """
    loop = asyncio.get_event_loop()
    try:
        with open(file_path, "rb") as f:
            resp = await loop.run_in_executor(
                None,
                lambda: openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            )
        return resp.text
    except Exception as e:
        print(f"[OpenAI] Ошибка при расшифровке аудио: {e}")
        return ""


async def generate_text(prompt: str,
                        model: str = "gpt-4o",
                        max_tokens: int = 2000) -> str:
    """
    Простая генерация текста через Chat Completions.
    """
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0
            )
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenAI] Ошибка при генерации текста: {e}")
        return "Ошибка при генерации ответа."


async def generate_analysis_text(prompt: str,
                                 model: str = "gpt-4o") -> str:
    """
    Генерация развёрнутого анализа (для daily_report).
    """
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenAI] Ошибка при генерации анализа: {e}")
        return ""
