# main.py
import uvicorn
import asyncio
import logging
import os

from fastapi import FastAPI
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
app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "message": "Bot is running"}


async def on_startup():
    # Инициализируем БД
    await init_db()

    # Создаём бота
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    bot.parse_mode = ParseMode.HTML

    # Подключаем роутер aiogram
    dp = Dispatcher()
    dp.include_router(router)

    # Запускаем бота (polling)
    asyncio.create_task(dp.start_polling(bot, skip_updates=True))
    logging.info("Бот запущен. Начался polling...")

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


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(on_startup())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
