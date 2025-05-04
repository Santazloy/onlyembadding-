# openai_utils.py
import openai
import asyncio

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


async def get_embedding(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: openai.Embedding.create(model="text-embedding-3-large",
                                            input=text))
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"[OpenAI] Ошибка при получении эмбеддинга: {e}")
        return []


async def transcribe_audio(file_path: str) -> str:
    loop = asyncio.get_event_loop()
    try:
        with open(file_path, "rb") as audio_file:
            response = await loop.run_in_executor(
                None, lambda: openai.Audio.transcribe("whisper-1", audio_file))
        # response["text"] – готовая расшифровка
        return response["text"]
    except Exception as e:
        print(f"[OpenAI] Ошибка при расшифровке аудио: {e}")
        return ""


async def generate_text(prompt: str, model="gpt-4o", max_tokens=2000) -> str:
    """
    Пример использования openai.Completion, если нужно.
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None, lambda: openai.Completion.create(model=model,
                                                   prompt=prompt,
                                                   max_tokens=max_tokens,
                                                   temperature=0.7,
                                                   top_p=1.0))
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"[OpenAI] Ошибка при генерации текста: {e}")
        return "Ошибка при генерации ответа."


# Отдельная функция для анализа (daily_report)
async def generate_analysis_text(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: openai.ChatCompletion.create(model="gpt-4o",
                                                 messages=[{
                                                     "role": "user",
                                                     "content": prompt
                                                 }],
                                                 max_tokens=2000,
                                                 temperature=0.7))
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка GPT: {e}")
        return ""
