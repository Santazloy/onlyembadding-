import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

from db import init_db
from config import TELEGRAM_BOT_TOKEN
from handlers import router
from daily_report import send_reports_for_all_groups
from mamasan import send_random_questions, handle_callback_query

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

    # Регистрируем обработчик callback запросов для интерактивных кнопок Мама сан
    @dp.callback_query()
    async def process_callback(callback_query: types.CallbackQuery):
        await handle_callback_query(callback_query)

    # Настраиваем планировщик
    scheduler = AsyncIOScheduler(timezone=pytz.timezone("Asia/Shanghai"))

    # Ежедневный отчет в полночь по Шанхайскому времени
    scheduler.add_job(
        send_reports_for_all_groups,
        'cron',
        hour=0,
        minute=0,
        args=[bot],
        name='Ежедневные отчеты'
    )

    # Рассылка от Мама сан в полдень по Шанхайскому времени
    scheduler.add_job(
        send_random_questions,
        'cron',
        hour=12,
        minute=0,
        args=[bot],
        name='Советы от Мама сан'
    )

    scheduler.start()
    logging.info("Планировщик запущен: отчеты в 00:00, советы в 12:00 (Шанхай)")

    # Удаляем webhook, чтобы можно было использовать polling
    await bot.delete_webhook(drop_pending_updates=True)
    await asyncio.sleep(1)

    logging.info("Бот запущен. Начинается polling...")

    try:
        await dp.start_polling(bot, skip_updates=True)
    finally:
        # Корректное завершение работы
        scheduler.shutdown()
        await bot.session.close()
        logging.info("Бот остановлен корректно")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем")
