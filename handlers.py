# handlers.py

import os
import html
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile

from config import GROUP_IDS
from db import (
    save_embedding, count_embeddings, save_message,
    find_similar_messages, get_user_statistics,
    update_user_metrics, get_user_metrics_history
)
from openai_utils import get_embedding, transcribe_audio, analyze_sentiment
from daily_report import send_daily_report, send_period_report
from exchange import convert_and_format
from web import process_web_command
from mamasan import (
    VOICE_TRIGGERS_EN, VOICE_TRIGGERS_RU,
    TEXT_TRIGGERS_EN, TEXT_TRIGGERS_RU,
    detect_language_of_trigger,
    generate_gpt_reply, send_voice_reply,
    send_random_questions,
    mama_san,  # Глобальный экземпляр класса MamaSan
    send_interactive_menu,  # Функция интерактивного меню
    detect_character_mention  # Детектор упоминаний персонажей
)
from taro_module import (
    manara_cards, TARO_FOLDER, draw_cards,
    get_card_interpretation, REVERSED_CHANCE
)

logger = logging.getLogger(__name__)
router = Router()

# Кэш для временного хранения данных о сообщениях
message_cache: Dict[int, Dict] = {}


@router.message(Command("test"))
async def cmd_test(message: types.Message):
    total = await count_embeddings()
    await message.answer(f"Общее количество эмбеддингов в БД: {total}")


@router.message(Command("report"))
async def cmd_report(message: types.Message):
    """Ежедневный отчет за последние 24 часа"""
    await send_daily_report(message.bot, message.chat.id)
    await message.answer("📊 Отчёт за последние 24 часа сформирован!")


@router.message(Command("three"))
async def cmd_three_days_report(message: types.Message):
    """Отчет за последние 72 часа"""
    await message.answer("⏳ Формирую отчёт за последние 72 часа...")
    await send_period_report(message.bot, message.chat.id, days=3)
    await message.answer("📊 Отчёт за последние 72 часа отправлен!")


@router.message(Command("week"))
async def cmd_week_report(message: types.Message):
    """Отчет за последние 7 дней"""
    await message.answer("⏳ Формирую отчёт за последнюю неделю...")
    await send_period_report(message.bot, message.chat.id, days=7)
    await message.answer("📊 Недельный отчёт отправлен!")


@router.message(Command("month"))
async def cmd_month_report(message: types.Message):
    """Отчет за последние 30 дней"""
    await message.answer("⏳ Формирую отчёт за последний месяц (это может занять время)...")
    await send_period_report(message.bot, message.chat.id, days=30)
    await message.answer("📊 Месячный отчёт отправлен!")


@router.message(Command("mama"))
async def cmd_mama_menu(message: types.Message):
    """Команда для вызова интерактивного меню Мама сан"""
    await send_interactive_menu(message)


@router.message(Command("stats"))
async def cmd_user_stats(message: types.Message):
    """Команда для просмотра личной статистики"""
    user_id = message.from_user.id
    group_id = message.chat.id

    # Получаем статистику за последние 7 дней
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)

    user_stats = await get_user_statistics(group_id, start_time, end_time)
    user_data = user_stats.get(user_id, {})

    if not user_data:
        await message.answer("У вас пока нет статистики в этом чате.")
        return

    # Получаем историю метрик
    metrics_history = await get_user_metrics_history(group_id, user_id, days=7)

    stats_text = f"""
📊 <b>Ваша статистика за последние 7 дней:</b>

👤 <b>Пользователь:</b> {user_data.get('user_name', 'Unknown')}
💬 <b>Сообщений:</b> {user_data.get('message_count', 0)}
📝 <b>Средняя длина:</b> {user_data.get('avg_message_length', 0):.0f} символов
🕐 <b>Активных часов:</b> {user_data.get('active_hours', 0)}
🧠 <b>Эмбеддингов:</b> {user_data.get('embedding_count', 0)}

<b>Динамика по дням:</b>
"""

    for metric in metrics_history[:7]:
        date_str = metric['metric_date'].strftime('%d.%m')
        msg_count = metric.get('message_count', 0)
        influence = metric.get('influence_score', 0)
        stats_text += f"📅 {date_str}: {msg_count} сообщ., влияние: {influence:.2f}\n"

    await message.answer(stats_text, parse_mode="HTML")


@router.message(Command("similar"))
async def cmd_find_similar(message: types.Message):
    """Поиск похожих сообщений"""
    # Получаем текст после команды
    text = message.text.replace('/similar', '').strip()

    if not text:
        await message.answer("Использование: /similar <текст для поиска>")
        return

    # Получаем эмбеддинг для поиска
    search_embedding = await get_embedding(text)
    if not search_embedding:
        await message.answer("Ошибка при создании эмбеддинга для поиска.")
        return

    # Ищем похожие сообщения
    similar_messages = await find_similar_messages(
        message.chat.id,
        search_embedding,
        threshold=0.7,
        limit=5
    )

    if not similar_messages:
        await message.answer("Похожих сообщений не найдено.")
        return

    response = "🔍 <b>Похожие сообщения:</b>\n\n"
    for i, msg in enumerate(similar_messages, 1):
        response += f"{i}. <b>{msg['user_name']}</b> ({msg['created_at'].strftime('%d.%m %H:%M')})\n"
        response += f"   📊 Схожесть: {msg['similarity']:.1%}\n"
        response += f"   💬 {html.escape(msg['text'][:100])}{'...' if len(msg['text']) > 100 else ''}\n\n"

    await message.answer(response, parse_mode="HTML")


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


async def process_message_embedding(message: types.Message, text: str,
                                    message_id: Optional[int] = None):
    """Обработка и сохранение эмбеддинга сообщения"""
    # Получаем эмбеддинг
    emb = await get_embedding(text)
    if not emb:
        logger.error(f"Ошибка получения эмбеддинга для текста: {text[:50]}...")
        return None

    # Сохраняем эмбеддинг
    await save_embedding(
        group_id=message.chat.id,
        user_id=message.from_user.id,
        embedding_vector=emb,
        message_id=message_id
    )

    # Анализируем sentiment если это текстовое сообщение
    if len(text) > 10:
        sentiment = await analyze_sentiment(text)

        # Обновляем метрики пользователя
        await update_user_metrics(
            group_id=message.chat.id,
            user_id=message.from_user.id,
            metric_date=datetime.utcnow(),
            metrics={
                'sentiment_score': sentiment,
                'message_count': 1  # Будет инкрементироваться в БД
            }
        )

    logger.info(f"Эмбеддинг для сообщения {message_id} успешно сохранён")
    return emb


@router.message(F.text)
async def handle_text_message(message: types.Message):
    if message.chat.id not in GROUP_IDS:
        return

    text = message.text.strip()
    if not text:
        return

    user_name = message.from_user.full_name if message.from_user else "unknown"

    # Определяем, является ли это ответом
    reply_to_id = None
    if message.reply_to_message:
        reply_to_id = message_cache.get(message.reply_to_message.message_id, {}).get('db_id')

    # Сохраняем сообщение
    message_db_id = await save_message(
        group_id=message.chat.id,
        user_id=message.from_user.id,
        user_name=user_name,
        text=text,
        reply_to_message_id=reply_to_id
    )

    # Кэшируем ID для последующих ответов
    message_cache[message.message_id] = {
        'db_id': message_db_id,
        'timestamp': datetime.utcnow()
    }

    # Очищаем старый кэш (старше 1 часа)
    await cleanup_message_cache()

    # Обрабатываем эмбеддинг
    await process_message_embedding(message, text, message_db_id)

    # Проверяем триггеры для ответа
    lang = detect_language_of_trigger(text)
    text_lower = text.lower()

    # Проверяем упоминание персонажей команды
    character_mention = detect_character_mention(text)
    if character_mention and lang:
        # Если упомянут персонаж, рассказываем историю о нём
        story = mama_san.get_character_story(character_mention)
        await message.answer(story)
        return

    # Обработка голосовых триггеров
    if any(t in text_lower for t in VOICE_TRIGGERS_EN + VOICE_TRIGGERS_RU):
        # Используем новый контекстный генератор ответов
        reply = await mama_san.generate_contextual_reply(
            text,
            message.from_user.id,
            user_name,
            lang
        )
        await send_voice_reply(message, reply, lang)
        return

    # Обработка текстовых триггеров
    if any(t in text_lower for t in TEXT_TRIGGERS_EN + TEXT_TRIGGERS_RU):
        # Используем новый контекстный генератор ответов
        reply = await mama_san.generate_contextual_reply(
            text,
            message.from_user.id,
            user_name,
            lang
        )
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

    # Сохраняем транскрибированное сообщение
    message_db_id = await save_message(
        group_id=message.chat.id,
        user_id=message.from_user.id,
        user_name=user_name,
        text=transcribed_text,
        message_type="voice"
    )

    # Отправляем транскрипцию
    await message.answer(f"🎤 {user_name}:\n{transcribed_text}")

    # Удаляем исходное voice сообщение
    try:
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
    except Exception as err:
        logger.warning("Не удалось удалить voice %s: %s", message.message_id, err)

    # Обрабатываем эмбеддинг
    await process_message_embedding(message, transcribed_text, message_db_id)

    # Проверяем триггеры в транскрибированном тексте
    lang = detect_language_of_trigger(transcribed_text)
    text_lower = transcribed_text.lower()

    # Проверяем упоминание персонажей
    character_mention = detect_character_mention(transcribed_text)
    if character_mention and lang:
        story = mama_san.get_character_story(character_mention)
        await message.answer(story)
        return

    # Если в голосовом сообщении есть триггеры, отвечаем
    if any(t in text_lower for t in VOICE_TRIGGERS_EN + VOICE_TRIGGERS_RU + TEXT_TRIGGERS_EN + TEXT_TRIGGERS_RU):
        reply = await mama_san.generate_contextual_reply(
            transcribed_text,
            message.from_user.id,
            user_name,
            lang if lang else "ru"  # По умолчанию русский
        )
        await send_voice_reply(message, reply, lang if lang else "ru")

    os.remove(local_filename)


async def cleanup_message_cache():
    """Очистка устаревших записей в кэше"""
    current_time = datetime.utcnow()
    cutoff_time = current_time - timedelta(hours=1)

    to_remove = []
    for msg_id, data in message_cache.items():
        if data['timestamp'] < cutoff_time:
            to_remove.append(msg_id)

    for msg_id in to_remove:
        del message_cache[msg_id]


@router.message()
async def debug_handler(message: types.Message):
    print("DEBUG:", message.chat.id, message.text)
