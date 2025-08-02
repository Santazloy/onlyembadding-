# mamasan.py

import os
import asyncio
import logging
import random
import subprocess
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum

from aiogram import Bot, types
from aiogram.enums import ChatAction
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from config import GROUP_IDS
from openai_utils import generate_text, analyze_sentiment, detect_emotions
from db import save_message, get_user_statistics

logger = logging.getLogger(__name__)


class MoodType(Enum):
    """Типы настроения Мама сан"""
    FRIENDLY = "friendly"  # Дружелюбная
    STRICT = "strict"  # Строгая
    CARING = "caring"  # Заботливая
    MOTIVATING = "motivating"  # Мотивирующая
    PLAYFUL = "playful"  # Игривая


class TopicCategory(Enum):
    """Категории вопросов"""
    BUSINESS = "Деловые встречи"
    LIFESTYLE = "Стиль жизни"
    BEAUTY = "Красота и уход"
    CULTURE = "Культура и этикет"
    PSYCHOLOGY = "Психология"
    PRACTICAL = "Практические советы"
    NETWORKING = "Нетворкинг"
    SAFETY = "Безопасность"


# Расширенные триггеры
VOICE_TRIGGERS_EN = ["lady", "lady san", "mama", "mama san", "madam"]
TEXT_TRIGGERS_EN = ["miss", "miss san", "ms", "ms san", "dear mama"]
VOICE_TRIGGERS_RU = ["леди", "леди сан", "мама", "мама сан", "мадам"]
TEXT_TRIGGERS_RU = ["мисс", "мисс сан", "дорогая мама", "мамочка"]

# Специальные триггеры для разных настроений
MOOD_TRIGGERS = {
    MoodType.STRICT: ["строго", "strictly", "наставление", "lecture"],
    MoodType.CARING: ["забота", "care", "поддержка", "support"],
    MoodType.MOTIVATING: ["мотивация", "motivation", "вдохнови", "inspire"],
    MoodType.PLAYFUL: ["пошути", "joke", "развесели", "fun"]
}

# Расширенный пул вопросов с категориями
CATEGORIZED_QUESTIONS = {
    TopicCategory.BUSINESS: [
        "Как правильно организовать встречу с VIP-клиентом в нашем агентстве?",
        "Какие требования нужно соблюсти при подготовке к встрече с высокопоставленным клиентом?",
        "Что учитывать при организации встречи с элитным клиентом?",
        "Как создать атмосферу эксклюзивности на деловой встрече?",
        "Какие детали помогут произвести впечатление профессионала высокого уровня?",
    ],
    TopicCategory.LIFESTYLE: [
        "Какие мероприятия помогут разнообразить досуг наших сотрудниц?",
        "Как найти баланс между работой и личной жизнью в мегаполисе?",
        "Какие хобби помогут развиваться профессионально и личностно?",
        "Как организовать идеальный выходной в Шанхае?",
        "Какие привычки успешных женщин стоит перенять?",
    ],
    TopicCategory.BEAUTY: [
        "Какие современные тренды в макияже ты рекомендуешь для деловой встречи?",
        "Как создать стильный образ для участия в гламурной вечеринке в Шанхае?",
        "Какие прически сейчас актуальны для эскорт-девушек?",
        "Секреты ухода за кожей в условиях мегаполиса?",
        "Как выбрать парфюм, который подчеркнет индивидуальность?",
    ],
    TopicCategory.CULTURE: [
        "Можешь привести примеры мандаринских фраз для приветствия клиента?",
        "Какие элементы китайского этикета важно соблюдать при общении с клиентами?",
        "Как правильно дарить и принимать подарки в китайской культуре?",
        "Какие темы разговора считаются табу в китайском обществе?",
        "Как показать уважение к китайским традициям во время делового ужина?",
    ],
    TopicCategory.PSYCHOLOGY: [
        "Какие методы помогают справляться со стрессом при напряжённом рабочем графике?",
        "Как сохранить эмоциональную стабильность в условиях высокой нагрузки?",
        "Техники быстрого восстановления энергии между встречами?",
        "Как развить эмоциональный интеллект для работы с разными клиентами?",
        "Способы поддержания позитивного настроя в сложных ситуациях?",
    ],
    TopicCategory.PRACTICAL: [
        "Как быстро разобраться с приобретением SIM-карты в Китае?",
        "Какие шаги нужно предпринять для открытия банковского счёта в Шанхае?",
        "Лучшие мобильные приложения для жизни в Шанхае?",
        "Как эффективно использовать WeChat для работы и жизни?",
        "Секреты экономии при шопинге в Шанхае?",
    ],
    TopicCategory.NETWORKING: [
        "Какие выставки и конференции в Шанхае помогут расширить деловые контакты?",
        "Как правильно обмениваться визитками в Китае?",
        "Где найти элитные клубы для нетворкинга в Шанхае?",
        "Как поддерживать профессиональные связи без навязчивости?",
        "Секреты успешного small talk на деловых мероприятиях?",
    ],
    TopicCategory.SAFETY: [
        "Как безопасно вести профиль в соцсетях для эскорт-девушек?",
        "Какие меры предосторожности соблюдать при первой встрече с клиентом?",
        "Как защитить личную информацию в цифровую эпоху?",
        "Что делать в экстренной ситуации во время встречи?",
        "Как выбрать безопасное место для деловой встречи?",
    ]
}

# Персонажи команды с расширенными характеристиками
TEAM_CHARACTERS = {
    "панда-егор": {
        "name": "Егор Панда",
        "role": "Главный по безопасности",
        "traits": ["спокойный", "надёжный", "заботливый"],
        "stories": [
            "Однажды Егор-панда уснул прямо во время важной встречи, но клиент подумал, что это медитация, и тоже начал медитировать!",
            "Егор всегда носит с собой термос с бамбуковым чаем и угощает им всех, кто выглядит уставшим.",
            "Когда Егор улыбается, все проблемы кажутся решаемыми - недаром его называют 'терапевтическая панда'!"
        ]
    },
    "гепард-лео": {
        "name": "Лео Гепард",
        "role": "Координатор встреч",
        "traits": ["быстрый", "энергичный", "харизматичный"],
        "stories": [
            "Лео однажды организовал три встречи одновременно в разных районах города и успел на все - правда, на последней был в тапочках!",
            "Гепард-Лео может найти любое место в Шанхае с закрытыми глазами, но постоянно путает левое и правое.",
            "Его рекорд - 47 сообщений в минуту. Девочки говорят, что у него клавиатура дымится!"
        ]
    },
    "медведь-михаил": {
        "name": "Михаил Медведь",
        "role": "Финансовый директор",
        "traits": ["основательный", "щедрый", "защитник"],
        "stories": [
            "Миша-медведь однажды перепутал юани с йенами и случайно дал девочкам тройную премию - теперь это наша традиция!",
            "Когда Михаил считает деньги, он рычит - говорит, это помогает концентрации.",
            "Медведь-Михаил коллекционирует калькуляторы и у него их уже 73 штуки!"
        ]
    },
    "белка-крис": {
        "name": "Крис Белка",
        "role": "PR-менеджер",
        "traits": ["креативная", "общительная", "изобретательная"],
        "stories": [
            "Крис-белка однажды организовала фотосессию на крыше небоскрёба и чуть не улетела с зонтиком!",
            "Она прячет орешки по всему офису 'на чёрный день' - уже нашли 47 тайников!",
            "Белка-Крис может превратить любую историю в вирусный пост - даже про то, как кто-то чихнул!"
        ]
    }
}


class MamaSan:
    """Класс для управления персонажем Мама сан"""

    def __init__(self):
        self.mood = MoodType.FRIENDLY
        self.conversation_history: Dict[int, List[Dict]] = defaultdict(list)
        self.user_preferences: Dict[int, Dict] = defaultdict(dict)
        self.last_interaction: Dict[int, datetime] = {}

    def detect_mood_from_text(self, text: str) -> MoodType:
        """Определение настроения из текста"""
        text_lower = text.lower()

        for mood, triggers in MOOD_TRIGGERS.items():
            if any(trigger in text_lower for trigger in triggers):
                return mood

        # Анализ эмоционального контекста
        if any(word in text_lower for word in ["помоги", "help", "проблема", "problem"]):
            return MoodType.CARING
        elif any(word in text_lower for word in ["устала", "tired", "сложно", "hard"]):
            return MoodType.MOTIVATING

        return MoodType.FRIENDLY

    def get_personalized_greeting(self, user_id: int, user_name: str) -> str:
        """Персонализированное приветствие"""
        last_seen = self.last_interaction.get(user_id)

        if not last_seen:
            return f"Рада познакомиться, {user_name}! Я Мама сан, и я здесь, чтобы помочь тебе."

        time_diff = datetime.now() - last_seen

        if time_diff.days > 7:
            return f"Давно не виделись, {user_name}! Как ты? Расскажи, что нового."
        elif time_diff.days > 1:
            return f"С возвращением, {user_name}! Надеюсь, у тебя всё хорошо."
        else:
            return f"Снова привет, {user_name}! Чем могу помочь?"

    def get_mood_prompt(self, mood: MoodType) -> str:
        """Получение промпта в зависимости от настроения"""
        mood_prompts = {
            MoodType.FRIENDLY: "Отвечай тепло и дружелюбно, с лёгким юмором.",
            MoodType.STRICT: "Будь строгой, но справедливой. Дай чёткие указания.",
            MoodType.CARING: "Покажи заботу и понимание. Поддержи и успокой.",
            MoodType.MOTIVATING: "Вдохнови и мотивируй! Покажи, что всё возможно.",
            MoodType.PLAYFUL: "Будь игривой и весёлой, добавь шутки и эмодзи."
        }
        return mood_prompts.get(mood, mood_prompts[MoodType.FRIENDLY])

    async def generate_contextual_reply(self, text: str, user_id: int,
                                        user_name: str, lang: str) -> str:
        """Генерация контекстного ответа"""
        mood = self.detect_mood_from_text(text)
        mood_instruction = self.get_mood_prompt(mood)

        # Анализ истории разговора
        history = self.conversation_history[user_id][-5:]  # Последние 5 сообщений
        context = "\n".join([f"{h['role']}: {h['content']}" for h in history])

        system_prompt = f"""Ты — Мама сан из эскорт-агентства YCF в Шанхае с 9-летним стажем.
        Текущее настроение: {mood_instruction}

        Помни:
        - Ты обращаешься к девушке по имени {user_name}
        - Учитывай контекст предыдущего разговора
        - Используй эмодзи для выразительности
        - Давай конкретные, практичные советы
        - Ссылайся на персонажей команды, если уместно

        Команда: {', '.join(TEAM_CHARACTERS.keys())}

        {"Reply in English." if lang == "en" else "Отвечай на русском языке."}
        """

        prompt = f"{system_prompt}\n\nКонтекст разговора:\n{context}\n\nВопрос: {text}"

        try:
            reply = await generate_text(
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=2000,
                temperature=0.8
            )

            # Сохраняем в историю
            self.conversation_history[user_id].append({
                "role": "user",
                "content": text,
                "timestamp": datetime.now()
            })
            self.conversation_history[user_id].append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now()
            })

            # Обновляем время последнего взаимодействия
            self.last_interaction[user_id] = datetime.now()

            return reply.strip()

        except Exception as e:
            logger.error(f"GPT error: {e}")
            return "Ой, что-то пошло не так, дорогая. Попробуй ещё раз! 💕"

    def get_character_story(self, character_key: str) -> str:
        """Получение истории о персонаже"""
        char = TEAM_CHARACTERS.get(character_key)
        if not char:
            return "Не знаю такого персонажа в нашей команде!"

        story = random.choice(char['stories'])
        traits = ", ".join(char['traits'])

        return f"О, {char['name']}! {char['role']} нашей команды. Он {traits}.\n\n{story} 😄"

    def get_daily_tips(self, category: Optional[TopicCategory] = None) -> List[str]:
        """Получение ежедневных советов"""
        if category:
            questions = CATEGORIZED_QUESTIONS.get(category, [])
        else:
            # Собираем вопросы из всех категорий
            questions = []
            for cat_questions in CATEGORIZED_QUESTIONS.values():
                questions.extend(cat_questions)

        # Выбираем случайные вопросы
        return random.sample(questions, min(5, len(questions)))


# Глобальный экземпляр
mama_san = MamaSan()


def detect_language_of_trigger(text: str) -> str:
    """Определение языка по триггерам"""
    txt_lower = text.lower()
    if any(t in txt_lower for t in VOICE_TRIGGERS_EN + TEXT_TRIGGERS_EN):
        return "en"
    if any(t in txt_lower for t in VOICE_TRIGGERS_RU + TEXT_TRIGGERS_RU):
        return "ru"
    return ""


def detect_character_mention(text: str) -> Optional[str]:
    """Определение упоминания персонажа"""
    text_lower = text.lower()
    for character in TEAM_CHARACTERS.keys():
        if character in text_lower or character.split('-')[1] in text_lower:
            return character
    return None


async def generate_gpt_reply(text: str, lang: str) -> str:
    """Устаревшая функция для обратной совместимости"""
    # Используем фиктивные данные для user_id и user_name
    return await mama_san.generate_contextual_reply(text, 0, "дорогая", lang)


async def send_voice_reply(message: types.Message, text: str, lang: str):
    """Отправка голосового сообщения с улучшенной обработкой"""
    TTS_API_KEY = os.getenv("TTS_API_KEY", "")
    TTS_API_ENDPOINT = os.getenv("TTS_API_ENDPOINT", "")
    FFMPEG_CMD = os.getenv("FFMPEG_PATH", "ffmpeg")

    if not TTS_API_KEY or not TTS_API_ENDPOINT:
        # Отправляем текст с эмодзи для голосового сообщения
        await message.answer(f"🎤 {text}")
        return

    try:
        # Показываем, что записываем голосовое
        await message.bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)

        # Выбираем голос в зависимости от настроения
        voice = "coral"  # Можно менять в зависимости от настроения

        payload = {
            "input": text,
            "voice": voice,
            "response_format": "opus",
            "model": "tts-1-hd"
        }
        headers = {
            "Authorization": f"Bearer {TTS_API_KEY}",
            "Content-Type": "application/json"
        }

        resp = requests.post(TTS_API_ENDPOINT, json=payload, headers=headers, timeout=30)

        if resp.status_code == 200:
            with open("raw.opus", "wb") as f:
                f.write(resp.content)

            # Конвертируем с оптимизацией
            subprocess.run([
                FFMPEG_CMD, "-y", "-i", "raw.opus",
                "-c:a", "libopus", "-ar", "48000",
                "-ac", "1", "-b:a", "64k",
                "-map_metadata", "-1",
                "-f", "ogg", "speech.ogg"
            ], check=True, capture_output=True)

            voice_file = FSInputFile("speech.ogg")
            await message.answer_voice(voice_file, caption="💕 Мама сан")

            # Удаляем временные файлы
            for f in ["raw.opus", "speech.ogg"]:
                if os.path.exists(f):
                    os.remove(f)

        else:
            logger.error(f"TTS API error: {resp.status_code}")
            await message.answer(f"🎤 {text}")

    except Exception as e:
        logger.error(f"TTS error: {e}")
        await message.answer(f"🎤 {text}")


async def send_interactive_menu(message: types.Message):
    """Отправка интерактивного меню"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="💼 Бизнес", callback_data="tips_business"),
            InlineKeyboardButton(text="💄 Красота", callback_data="tips_beauty")
        ],
        [
            InlineKeyboardButton(text="🧘 Психология", callback_data="tips_psychology"),
            InlineKeyboardButton(text="🌏 Культура", callback_data="tips_culture")
        ],
        [
            InlineKeyboardButton(text="🔧 Практика", callback_data="tips_practical"),
            InlineKeyboardButton(text="🤝 Нетворкинг", callback_data="tips_networking")
        ],
        [
            InlineKeyboardButton(text="🎲 Случайные советы", callback_data="tips_random")
        ]
    ])

    await message.answer(
        "🌸 <b>Выбери категорию советов, дорогая:</b>",
        reply_markup=keyboard,
        parse_mode="HTML"
    )


async def send_random_questions(bot: Bot):
    """Улучшенная рассылка вопросов с категориями и статистикой"""
    # Анализируем активность за последние 24 часа
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)

    # Формируем персонализированное приветствие
    current_hour = datetime.now().hour

    if 6 <= current_hour < 12:
        greeting = "🌅 Доброе утро, мои дорогие!"
    elif 12 <= current_hour < 18:
        greeting = "☀️ Добрый день, красавицы!"
    elif 18 <= current_hour < 23:
        greeting = "🌆 Добрый вечер, милые!"
    else:
        greeting = "🌙 Доброй ночи, девочки!"

    # Выбираем категорию дня
    category_of_day = random.choice(list(TopicCategory))
    category_questions = mama_san.get_daily_tips(category_of_day)

    # Добавляем интересный факт или историю
    character = random.choice(list(TEAM_CHARACTERS.keys()))
    char_story = mama_san.get_character_story(character)

    # Формируем сообщение
    message = f"""
{greeting}

<b>🎭 Я Леди Сан, ваша Мама сан!</b>

<b>📚 Категория дня: {category_of_day.value}</b>

<b>💡 Вопросы для размышления:</b>
{chr(10).join(f'• <i>{q}</i>' for q in category_questions[:3])}

<b>😄 История дня:</b>
{char_story}

<b>🗣 Как со мной общаться:</b>
🎤 Голосовой ответ: <code>леди</code>, <code>мама сан</code>
💬 Текстовый ответ: <code>мисс</code>, <code>дорогая мама</code>

<b>✨ Совет дня:</b>
<i>Помни, дорогая: ты не просто красива, ты профессионал высокого уровня! 
Каждый день - это новая возможность стать лучше.</i>

Обнимаю, ваша Мама сан 💕
"""

    # Добавляем интерактивные кнопки
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📝 Больше советов", callback_data="more_tips"),
            InlineKeyboardButton(text="❓ Задать вопрос", callback_data="ask_mama")
        ]
    ])

    # Отправляем в группы
    for chat_id in GROUP_IDS:
        try:
            await bot.send_message(
                chat_id,
                message,
                parse_mode="HTML",
                reply_markup=keyboard
            )

            # Логируем отправку
            logger.info(f"Daily tips sent to group {chat_id}")

        except Exception as e:
            logger.error(f"Error sending daily tips to {chat_id}: {e}")


async def handle_callback_query(callback_query: types.CallbackQuery):
    """Обработка callback запросов"""
    data = callback_query.data

    if data.startswith("tips_"):
        category_map = {
            "tips_business": TopicCategory.BUSINESS,
            "tips_beauty": TopicCategory.BEAUTY,
            "tips_psychology": TopicCategory.PSYCHOLOGY,
            "tips_culture": TopicCategory.CULTURE,
            "tips_practical": TopicCategory.PRACTICAL,
            "tips_networking": TopicCategory.NETWORKING,
            "tips_random": None
        }

        category = category_map.get(data)
        tips = mama_san.get_daily_tips(category)

        tips_text = "\n\n".join(f"💭 {tip}" for tip in tips)
        category_name = category.value if category else "Случайные советы"

        await callback_query.message.edit_text(
            f"<b>📌 {category_name}:</b>\n\n{tips_text}\n\n"
            f"<i>С любовью, Мама сан 💕</i>",
            parse_mode="HTML"
        )

    elif data == "more_tips":
        await send_interactive_menu(callback_query.message)

    elif data == "ask_mama":
        await callback_query.message.answer(
            "Задай мне вопрос, начав сообщение с 'Мама сан' или 'Леди' 💬"
        )

    await callback_query.answer()


# Функция для анализа эффективности советов
async def analyze_tips_effectiveness(bot: Bot, group_id: int):
    """Анализ эффективности советов на основе реакций"""
    # Здесь можно добавить логику анализа реакций на советы
    # и корректировку будущих рекомендаций
    pass
