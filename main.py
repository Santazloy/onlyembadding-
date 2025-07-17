# main.py
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

from db import init_db
from config import TELEGRAM_BOT_TOKEN
from handlers import router
from daily_report import send_reports_for_all_groups
from mamasan import send_random_questions

logging.basicConfig(level=logging.INFO)


async def main():
    # Инициализируем БД
    await init_db()

    # Создаём бота
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    bot.parse_mode = ParseMode.HTML

    # Подключаем роутер aiogram
    dp = Dispatcher()
    dp.include_router(router)

    # Создаём планировщик
    scheduler = AsyncIOScheduler(timezone=pytz.timezone("Asia/Shanghai"))
    # Ежедневный отчёт /report
    scheduler.add_job(send_reports_for_all_groups,
                      'cron',
                      hour=0,
                      minute=0,
                      args=[bot])
    # Рассылка случайных вопросов (в 12:00)
    scheduler.add_job(send_random_questions,
                      'cron',
                      hour=12,
                      minute=0,
                      args=[bot])

    scheduler.start()

    logging.info("Бот запущен. Начинается polling...")
    # Запускаем бота и не выходим, пока бот работает
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен")
