import openai
import asyncio

from config import OPENAI_API_KEY

# Инициализация ключа OpenAI
openai.api_key = OPENAI_API_KEY

async def get_embedding(text: str) -> list[float]:
    """
    Возвращает векторное представление текста с использованием модели text-embedding-3-large.
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"[OpenAI] Ошибка при получении эмбеддинга: {e}")
        return []

async def transcribe_audio(file_path: str) -> str:
    """
    Транскрибирует аудио-файл через модель whisper-1.
    """
    loop = asyncio.get_event_loop()
    try:
        with open(file_path, "rb") as audio_file:
            response = await loop.run_in_executor(
                None,
                lambda: openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            )
        return response["text"]
    except Exception as e:
        print(f"[OpenAI] Ошибка при расшифровке аудио: {e}")
        return ""

async def generate_text(prompt: str, model: str = "gpt-4o", max_tokens: int = 2000) -> str:
    """
    Генерация текста через GPT-4o.
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenAI] Ошибка при генерации текста: {e}")
        return "Ошибка при генерации ответа."

async def generate_analysis_text(prompt: str, model: str = "gpt-4o") -> str:
    """
    Генерация развёрнутого анализа (daily_report) через GPT-4o.
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenAI] Ошибка при генерации анализа: {e}")
        return ""
