# handlers.py
from aiogram import Router, types, F
from aiogram.filters import Command
from db import save_embedding, count_embeddings
from openai_utils import get_embedding
from config import GROUP_IDS
import logging
logger = logging.getLogger(__name__)

router = Router()

@router.message(Command("test"))
async def cmd_test(message: types.Message):
    """Команда /test — показать количество эмбеддингов."""
    total = await count_embeddings()
    await message.answer(f"Общее количество эмбеддингов в БД: {total}")


@router.message(F.text)
async def handle_text_message(message: types.Message):
    if message.chat.id not in GROUP_IDS:
        return

    text = message.text.strip()
    if not text:
        return

    embedding = await get_embedding(text)
    if embedding:
        # Сохраняем эмбеддинг
        await save_embedding(message.chat.id, message.from_user.id, embedding)
        logger.info(f"Эмбеддинг для сообщения: '{text}' успешно сохранён.")  # Логируем успех
    else:
        logger.error(f"Ошибка при получении эмбеддинга для сообщения: '{text}'")  # Логируем ошибку


# Если нужно обрабатывать голосовые, например:
@router.message(F.voice)
async def handle_voice_message(message: types.Message):
    """Обрабатываем голосовые сообщения (пришли из одной из групп)."""
    if message.chat.id not in GROUP_IDS:
        return

    # 1) Скачиваем voice -> transcribe -> ...
    # 2) Получаем текст -> get_embedding и т.д.
    # и т.п.
    await message.answer("Я пока не умею обрабатывать голосовые!")