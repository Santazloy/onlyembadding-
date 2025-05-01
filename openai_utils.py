# openai_utils.py
import openai
import asyncio

from config import OPENAI_API_KEY

# Устанавливаем API-ключ для openai
openai.api_key = OPENAI_API_KEY

async def get_embedding(text: str) -> list[float]:
    """
    Получить эмбеддинг для текста с помощью OpenAI API.
    Модель: text-embedding-3-large (убедитесь, что у вас есть доступ!)
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: openai.Embedding.create(
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
    Пример: расшифровать аудио (голосовое сообщение) с помощью модели Whisper.
    Модель: whisper-1
    """
    loop = asyncio.get_event_loop()
    try:
        audio_file = open(file_path, "rb")
        response = await loop.run_in_executor(
            None,
            lambda: openai.Audio.transcribe("whisper-1", audio_file)
        )
        # response["text"] – готовая расшифровка
        return response["text"]
    except Exception as e:
        print(f"[OpenAI] Ошибка при расшифровке аудио: {e}")
        return ""
    finally:
        audio_file.close()


async def generate_text(prompt: str, model="gpt-4o", max_tokens=1024) -> str:
    """
    Генерация текста с помощью GPT-модели.
    """
    loop = asyncio.get_event_loop()
    try:
        # Или openai.ChatCompletion.create() для ChatGPT-стиля
        response = await loop.run_in_executor(
            None,
            lambda: openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0
            )
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"[OpenAI] Ошибка при генерации текста: {e}")
        return "Ошибка при генерации ответа."