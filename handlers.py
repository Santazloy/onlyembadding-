import os
import html
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
from mamasan import (
    VOICE_TRIGGERS_EN, VOICE_TRIGGERS_RU,
    TEXT_TRIGGERS_EN, TEXT_TRIGGERS_RU,
    detect_language_of_trigger,
    generate_gpt_reply, send_voice_reply,
    send_random_questions
)
from taro_module import (
    manara_cards, TARO_FOLDER, draw_cards,
    get_card_interpretation, REVERSED_CHANCE
)

logger = logging.getLogger(__name__)
router = Router()


@router.message(Command("test"))
async def cmd_test(message: types.Message):
    total = await count_embeddings()
    await message.answer(f"Общее количество эмбеддингов в БД: {total}")


@router.message(Command("report"))
async def cmd_report(message: types.Message):
    await send_daily_report(message.bot, message.chat.id)
    await message.answer("Отчёт сформирован для этого чата!")


@router.message(Command("ex"))
async def ex_command_handler(message: types.Message):
    parts = message.text.split()
    args = parts[1:]
    if len(args) != 2:
        return await message.answer(
            "❌ Использование: /ex <сумма> <валюта>. Пример: /ex 100 USD"
        )
    try:
        amount = float(args[0])
        base_currency = args[1]
    except ValueError:
        return await message.answer("❌ Неверное значение суммы.")

    result_text = convert_and_format(amount, base_currency)
    await message.answer(result_text, parse_mode="HTML")


@router.message(Command("taro"))
async def cmd_taro(message: types.Message):
    try:
        cards = draw_cards(manara_cards, count=3, reversed_chance=REVERSED_CHANCE)
    except ValueError as e:
        return await message.answer(f"Ошибка: {e}")

    positions = ["Прошлое", "Настоящее", "Будущее"]
    for i, (fname, card_name, is_rev) in enumerate(cards):
        pos_text = positions[i]
        orientation = " (перевёрнутая)" if is_rev else ""
        caption = f"{pos_text}: {card_name}{orientation}"
        full_path = os.path.join(TARO_FOLDER, fname)

        interpretation = await get_card_interpretation(card_name, pos_text, is_rev)

        try:
            photo = FSInputFile(full_path)
            await message.answer_photo(photo=photo, caption=caption)
        except FileNotFoundError:
            await message.answer(
                f"**Не найден файл изображения**: {full_path}\n\n"
                f"{caption}\n\n{interpretation}",
                parse_mode="Markdown"
            )
            continue

        await message.answer(f"<pre>{html.escape(interpretation)}</pre>",
                             parse_mode="HTML")


@router.message(Command("web"))
async def cmd_web(message: types.Message):
    await process_web_command(message)


@router.message(F.text)
async def handle_text_message(message: types.Message):
    if message.chat.id not in GROUP_IDS:
        return

    text = message.text.strip()
    if not text:
        return

    user_name = message.from_user.full_name if message.from_user else "unknown"
    await save_message(
        group_id=message.chat.id,
        user_id=message.from_user.id,
        user_name=user_name,
        text=text
    )

    emb = await get_embedding(text)
    if emb:
        await save_embedding(message.chat.id, message.from_user.id, emb)
        logger.info(f"Эмбеддинг (text) для '{text}' успешно сохранён.")
    else:
        logger.error(f"Ошибка эмбеддинга (text) для '{text}'")

    lang = detect_language_of_trigger(text)
    text_lower = text.lower()

    if any(t in text_lower for t in VOICE_TRIGGERS_EN + VOICE_TRIGGERS_RU):
        reply = await generate_gpt_reply(text_lower, lang)
        await send_voice_reply(message, reply, lang)
        return

    if any(t in text_lower for t in TEXT_TRIGGERS_EN + TEXT_TRIGGERS_RU):
        reply = await generate_gpt_reply(text_lower, lang)
        await message.answer(reply)
        return


@router.message(F.voice)
async def handle_voice_message(message: types.Message):
    if message.chat.id not in GROUP_IDS:
        return

    bot = message.bot
    file_info = await bot.get_file(message.voice.file_id)
    local_filename = "temp_voice.ogg"
    await bot.download_file(file_info.file_path, local_filename)

    transcribed_text = await transcribe_audio(local_filename)

    if not transcribed_text:
        os.remove(local_filename)
        return await message.answer("Не удалось распознать голосовое сообщение :(")

    user_name = message.from_user.full_name if message.from_user else "unknown"
    await save_message(
        message.chat.id,
        message.from_user.id,
        user_name,
        transcribed_text
    )

    emb = await get_embedding(transcribed_text)
    await message.answer(f"{user_name}:\n{transcribed_text}")

    # удаляем исходное voice сообщение
    try:
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
    except Exception as err:
        logger.warning("Не удалось удалить voice %s: %s", message.message_id, err)

    if emb:
        await save_embedding(message.chat.id, message.from_user.id, emb)
    else:
        await message.answer("Ошибка при получении эмбеддинга.")

    os.remove(local_filename)


@router.message()
async def debug_handler(message: types.Message):
    print("DEBUG:", message.chat.id, message.text)
