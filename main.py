# main.py
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from aiogram.enums import ParseMode

from db import init_db
from config import TELEGRAM_BOT_TOKEN
from handlers import router

logging.basicConfig(level=logging.INFO)

async def main():
    await init_db()

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    bot.parse_mode = ParseMode.HTML

    dp = Dispatcher()
    dp.include_router(router)

    # Если вы объявляли какие-то команды - оставьте
    await bot.set_my_commands([
        BotCommand(command="test", description="Показать количество эмбеддингов")
    ])

    logging.info("Бот запущен. Начинаем поллинг...")
    # `skip_updates=True` — пропускает все старые сообщения
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())