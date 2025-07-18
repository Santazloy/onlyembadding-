# handlers.py
import os
import logging
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile

from config import GROUP_IDS
from db import save_embedding, count_embeddings, save_message
from openai_utils import get_embedding, transcribe_audio
from daily_report import send_daily_report
from exchange import convert_and_format
from web import process_web_command
# Импортируем «триггеры» и функции для GPT/TTS
# Теперь у нас есть разделение триггеров на английские/русские,
# detect_language_of_trigger(...) и т.д.
from mamasan import (VOICE_TRIGGERS_EN, VOICE_TRIGGERS_RU, TEXT_TRIGGERS_EN,
                     TEXT_TRIGGERS_RU, detect_language_of_trigger,
                     generate_gpt_reply, send_voice_reply,
                     send_random_questions)

# Таро
from taro_module import (manara_cards, TARO_FOLDER, draw_cards,
                         get_card_interpretation, REVERSED_CHANCE)

logger = logging.getLogger(__name__)
router = Router()


#
# 1) Команды
#
@router.message(Command("test"))
async def cmd_test(message: types.Message):
    """
    Показывает количество эмбеддингов в таблице embeddings.
    """
    total = await count_embeddings()
    await message.answer(f"Общее количество эмбеддингов в БД: {total}")


@router.message(Command("report"))
async def cmd_report(message: types.Message):
    """
    Вызывает формирование и отправку отчёта (daily_report.py)
    для текущего чата.
    """
    bot = message.bot
    await send_daily_report(bot, message.chat.id)
    await message.answer("Отчёт сформирован для этого чата!")


@router.message(Command("ex"))
async def ex_command_handler(message: types.Message):
    """
    Команда /ex <сумма> <валюта>,
    например: /ex 100 USD
    Обращается к convert_and_format из exchange.py
    """
    parts = message.text.split()
    args = parts[1:]
    if len(args) != 2:
        await message.answer(
            "❌ Использование: /ex <сумма> <валюта>. Пример: /ex 100 USD")
        return
    try:
        amount = float(args[0])
        base_currency = args[1]
    except ValueError:
        await message.answer("❌ Неверное значение суммы.")
        return

    result_text = convert_and_format(amount, base_currency)
    await message.answer(result_text, parse_mode="HTML")


@router.message(Command("taro"))
async def cmd_taro(message: types.Message):
    """
    Вытягиваем 3 карты из Таро Манары,
    отправляем фото + интерпретацию GPT.
    """
    try:
        cards = draw_cards(manara_cards,
                           count=3,
                           reversed_chance=REVERSED_CHANCE)
    except ValueError as e:
        await message.answer(f"Ошибка: {e}")
        return

    positions = ["Прошлое", "Настоящее", "Будущее"]
    for i, (fname, card_name, is_rev) in enumerate(cards):
        pos_text = positions[i]
        orientation = " (перевёрнутая)" if is_rev else ""
        caption = f"{pos_text}: {card_name}{orientation}"
        full_path = os.path.join(TARO_FOLDER, fname)

        interpretation = get_card_interpretation(card_name, pos_text, is_rev)

        # Пробуем отправить фото:
        try:
            photo = FSInputFile(full_path)
            await message.answer_photo(photo=photo, caption=caption)
        except FileNotFoundError:
            await message.answer(
                f"**Не найден файл изображения**: {full_path}\n\n"
                f"{caption}\n\n{interpretation}",
                parse_mode="Markdown")
            continue

        # Текст интерпретации (оформляем в <pre>)
        await message.answer(f"<pre>{interpretation}</pre>", parse_mode="HTML")


@router.message(Command("web"))
async def cmd_web(message: types.Message):
    await process_web_command(message)


#
# 2) Основной текстовый хендлер
#
@router.message(F.text)
async def handle_text_message(message: types.Message):
    """
    1) Сохраняем текст в БД + эмбеддинг (только если chat.id в GROUP_IDS)
    2) Проверяем триггеры:
       - если английский/русский VOICE => GPT + TTS
       - если английский/русский TEXT  => GPT (текст)
       - иначе — просто обычное сообщение без GPT
    """
    # Фильтр по ID групп:
    if message.chat.id not in GROUP_IDS:
        return

    text = message.text.strip()
    if not text:
        return

    # 1) Сохраняем в messages + embeddings
    user_name = message.from_user.full_name if message.from_user else "unknown"
    await save_message(group_id=message.chat.id,
                       user_id=message.from_user.id,
                       user_name=user_name,
                       text=text)
    emb = await get_embedding(text)
    if emb:
        await save_embedding(message.chat.id, message.from_user.id, emb)
        logger.info(f"Эмбеддинг (text) для '{text}' успешно сохранён.")
    else:
        logger.error(f"Ошибка эмбеддинга (text) для '{text}'")

    # 2) Ищем триггеры: определяем язык
    lang = detect_language_of_trigger(text)
    text_lower = text.lower()

    # Если найден триггер в voice (EN или RU)
    if any(t in text_lower for t in VOICE_TRIGGERS_EN + VOICE_TRIGGERS_RU):
        # Вызываем GPT + озвучка
        reply = await generate_gpt_reply(text_lower, lang)
        await send_voice_reply(message, reply, lang)
        return

    # Если найден триггер в text (EN или RU)
    if any(t in text_lower for t in TEXT_TRIGGERS_EN + TEXT_TRIGGERS_RU):
        reply = await generate_gpt_reply(text_lower, lang)
        await message.answer(reply)
        return


# 3) Голосовой хендлер
#
@router.message(F.voice)
async def handle_voice_message(message: types.Message):
    """
    1) Скачиваем voice -> temp_voice.ogg
    2) Расшифровываем (Whisper)
    3) Сохраняем -> embeddings
    """
    if message.chat.id not in GROUP_IDS:
        return

    bot = message.bot
    file_info = await bot.get_file(message.voice.file_id)
    local_filename = "temp_voice.ogg"
    await bot.download_file(file_info.file_path, local_filename)

    # Расшифровка
    transcribed_text = await transcribe_audio(local_filename)
    if not transcribed_text:
        await message.answer("Не удалось распознать голосовое сообщение :(")
        return

    # Сохраняем
    user_name = message.from_user.full_name if message.from_user else "unknown"
    await save_message(message.chat.id, message.from_user.id, user_name,
                       transcribed_text)

    emb = await get_embedding(transcribed_text)
    if emb:
        await save_embedding(message.chat.id, message.from_user.id, emb)
        await message.answer(f"Распознанный текст:\n{transcribed_text}")
    else:
        await message.answer("Ошибка при получении эмбеддинга.")


#
# 4) Хендлер отладки для всего прочего
#
@router.message()
async def debug_handler(message: types.Message):
    print("DEBUG:", message.chat.id, message.text)
