import re
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import asyncio
from aiogram import types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai_utils import generate_text, markdown_links_to_html

# Настройки
OPENAI_MODEL_WEBSEARCH = "gpt-4o-search-preview"
MAX_MESSAGE_LENGTH = 4096  # Telegram limit
MAX_RETRIES = 3
RETRY_DELAY = 1.0
RATE_LIMIT_WINDOW = 60  # секунд
RATE_LIMIT_MAX_REQUESTS = 10  # максимум запросов в окно

# Логирование
logger = logging.getLogger(__name__)


class RateLimiter:
    """Простой rate limiter для защиты от спама"""

    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS,
                 window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[int, deque] = {}  # user_id -> deque of timestamps

    def is_allowed(self, user_id: int) -> bool:
        """Проверить, можно ли выполнить запрос"""
        now = datetime.now()

        if user_id not in self.requests:
            self.requests[user_id] = deque()

        # Удаляем старые запросы
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.requests[user_id] and self.requests[user_id][0] < cutoff:
            self.requests[user_id].popleft()

        # Проверяем лимит
        if len(self.requests[user_id]) >= self.max_requests:
            return False

        # Добавляем текущий запрос
        self.requests[user_id].append(now)
        return True

    def get_wait_time(self, user_id: int) -> int:
        """Получить время ожидания в секундах до следующего разрешённого запроса"""
        if user_id not in self.requests or not self.requests[user_id]:
            return 0

        oldest_request = self.requests[user_id][0]
        wait_until = oldest_request + timedelta(seconds=self.window_seconds)
        wait_seconds = (wait_until - datetime.now()).total_seconds()

        return max(0, int(wait_seconds))


class HTMLSanitizer:
    """Класс для безопасной обработки HTML"""

    # Разрешённые теги и их атрибуты
    ALLOWED_TAGS = {
        'a': ['href'],
        'b': [],
        'i': [],
        'u': [],
        's': [],
        'code': [],
        'pre': [],
        'strong': [],
        'em': [],
        'blockquote': []
    }

    @staticmethod
    def sanitize(text: str) -> str:
        """
        Санитизация HTML с сохранением разрешённых тегов
        """
        # Сохраняем разрешённые теги
        placeholders = {}
        placeholder_count = 0

        # Находим все теги
        tag_pattern = re.compile(r'<(/?)(\w+)([^>]*)>', re.IGNORECASE)

        def replace_tag(match):
            nonlocal placeholder_count
            closing = match.group(1)
            tag_name = match.group(2).lower()
            attributes = match.group(3)

            if tag_name in HTMLSanitizer.ALLOWED_TAGS:
                # Проверяем атрибуты
                if attributes and not closing:
                    cleaned_attrs = HTMLSanitizer._clean_attributes(tag_name, attributes)
                    if cleaned_attrs:
                        full_tag = f'<{tag_name}{cleaned_attrs}>'
                    else:
                        full_tag = f'<{tag_name}>'
                else:
                    full_tag = f'<{closing}{tag_name}>'

                placeholder = f'__TAG_{placeholder_count}__'
                placeholders[placeholder] = full_tag
                placeholder_count += 1
                return placeholder
            else:
                # Неразрешённый тег - экранируем
                return f'&lt;{closing}{tag_name}{attributes}&gt;'

        # Заменяем теги на плейсхолдеры
        text = tag_pattern.sub(replace_tag, text)

        # Экранируем специальные символы
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')

        # Возвращаем разрешённые теги
        for placeholder, tag in placeholders.items():
            text = text.replace(placeholder, tag)

        return text

    @staticmethod
    def _clean_attributes(tag_name: str, attributes: str) -> str:
        """Очистка атрибутов тега"""
        allowed_attrs = HTMLSanitizer.ALLOWED_TAGS.get(tag_name, [])
        if not allowed_attrs:
            return ''

        cleaned = []
        # Простой парсинг атрибутов
        attr_pattern = re.compile(r'(\w+)=["\']([^"\']+)["\']')

        for match in attr_pattern.finditer(attributes):
            attr_name = match.group(1).lower()
            attr_value = match.group(2)

            if attr_name in allowed_attrs:
                # Дополнительная проверка для href
                if attr_name == 'href' and not HTMLSanitizer._is_safe_url(attr_value):
                    continue
                cleaned.append(f'{attr_name}="{attr_value}"')

        return ' ' + ' '.join(cleaned) if cleaned else ''

    @staticmethod
    def _is_safe_url(url: str) -> bool:
        """Проверка безопасности URL"""
        # Разрешаем только http/https и относительные ссылки
        url_lower = url.lower().strip()
        return (url_lower.startswith('http://') or
                url_lower.startswith('https://') or
                url_lower.startswith('/') or
                not any(url_lower.startswith(p) for p in ['javascript:', 'data:', 'vbscript:']))


class WebSearchHandler:
    """Обработчик команд веб-поиска"""

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.sanitizer = HTMLSanitizer()

    async def process_web_command(self, message: types.Message) -> None:
        """
        Обработка команды /web
        """
        # Проверка rate limit
        if not self.rate_limiter.is_allowed(message.from_user.id):
            wait_time = self.rate_limiter.get_wait_time(message.from_user.id)
            await message.answer(
                f"⏳ Слишком много запросов. Подождите {wait_time} секунд."
            )
            return

        # Парсинг команды
        query = self._extract_query(message.text)
        if not query:
            await self._send_help(message)
            return

        # Отправляем индикатор загрузки
        loading_msg = await message.answer("🔍 Ищу информацию...")

        try:
            # Выполняем поиск с повторными попытками
            result = await self._perform_search(query)

            # Обрабатываем и отправляем результат
            await self._send_result(message, result, loading_msg)

            # Логируем успешный запрос
            logger.info(f"Web search completed for user {message.from_user.id}: {query[:50]}...")

        except Exception as e:
            logger.error(f"Web search error for user {message.from_user.id}: {e}")
            await loading_msg.edit_text(
                "❌ Произошла ошибка при выполнении поиска. Попробуйте позже."
            )

    def _extract_query(self, text: str) -> Optional[str]:
        """Извлечь запрос из текста команды"""
        parts = text.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            return None
        return parts[1].strip()

    async def _send_help(self, message: types.Message) -> None:
        """Отправить справку по использованию"""
        help_text = (
            "🔍 <b>Использование веб-поиска:</b>\n\n"
            "<code>/web запрос</code> - поиск информации в интернете\n\n"
            "<b>Примеры:</b>\n"
            "• /web погода в Москве\n"
            "• /web последние новости AI\n"
            "• /web как приготовить пасту карбонара\n\n"
            "<i>Подсказка: формулируйте запросы конкретно для лучших результатов</i>"
        )
        await message.answer(help_text, parse_mode="HTML")

    async def _perform_search(self, query: str) -> str:
        """Выполнить поиск с повторными попытками"""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                # Улучшаем запрос для лучших результатов
                enhanced_query = self._enhance_query(query)

                result = await generate_text(
                    prompt=enhanced_query,
                    model=OPENAI_MODEL_WEBSEARCH
                )

                if result and result.strip():
                    return result
                else:
                    raise ValueError("Получен пустой ответ")

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    logger.warning(f"Retry {attempt + 1} for query: {query[:50]}...")

        raise last_error or Exception("Не удалось выполнить поиск")

    def _enhance_query(self, query: str) -> str:
        """Улучшить запрос для более точных результатов"""
        # Добавляем контекст для улучшения результатов
        enhancements = []

        # Определяем язык запроса
        if any(ord(c) > 127 for c in query):  # Не-ASCII символы
            if any('\u0400' <= c <= '\u04FF' for c in query):  # Кириллица
                enhancements.append("Отвечай на русском языке.")

        # Добавляем временной контекст для актуальных запросов
        time_keywords = ['сегодня', 'сейчас', 'последние', 'новости', 'актуальн']
        if any(keyword in query.lower() for keyword in time_keywords):
            enhancements.append(f"Сегодня {datetime.now().strftime('%d %B %Y года')}.")

        # Формируем улучшенный запрос
        enhanced = query
        if enhancements:
            enhanced = f"{query}\n\n{' '.join(enhancements)}"

        return enhanced

    async def _send_result(self, message: types.Message, result: str,
                           loading_msg: types.Message) -> None:
        """Отправить результат поиска"""
        # Конвертируем Markdown в HTML
        html_result = markdown_links_to_html(result)

        # Санитизируем HTML
        safe_html = self.sanitizer.sanitize(html_result)

        # Разбиваем на части если слишком длинный
        if len(safe_html) <= MAX_MESSAGE_LENGTH:
            try:
                await loading_msg.edit_text(
                    safe_html,
                    parse_mode="HTML",
                    disable_web_page_preview=False
                )
            except Exception as e:
                # Fallback на plain text
                logger.error(f"Failed to send HTML: {e}")
                await loading_msg.edit_text(
                    self._strip_html(result),
                    disable_web_page_preview=True
                )
        else:
            # Отправляем по частям
            await loading_msg.delete()
            parts = self._split_message(safe_html)

            for i, part in enumerate(parts):
                try:
                    await message.answer(
                        part,
                        parse_mode="HTML",
                        disable_web_page_preview=(i > 0)  # Preview только для первой части
                    )
                    if i < len(parts) - 1:
                        await asyncio.sleep(0.5)  # Небольшая задержка между сообщениями
                except Exception as e:
                    logger.error(f"Failed to send part {i + 1}: {e}")
                    await message.answer(
                        self._strip_html(part),
                        disable_web_page_preview=True
                    )

    def _split_message(self, text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
        """Разбить длинное сообщение на части"""
        if len(text) <= max_length:
            return [text]

        parts = []
        current_part = ""

        # Пытаемся разбить по параграфам
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if len(current_part) + len(para) + 2 <= max_length:
                if current_part:
                    current_part += '\n\n'
                current_part += para
            else:
                if current_part:
                    parts.append(current_part)

                # Если параграф слишком длинный, разбиваем по предложениям
                if len(para) > max_length:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_part = ""

                    for sent in sentences:
                        if len(current_part) + len(sent) + 1 <= max_length:
                            if current_part:
                                current_part += ' '
                            current_part += sent
                        else:
                            if current_part:
                                parts.append(current_part)
                            current_part = sent
                else:
                    current_part = para

        if current_part:
            parts.append(current_part)

        return parts

    def _strip_html(self, text: str) -> str:
        """Удалить все HTML теги для fallback"""
        return re.sub(r'<[^>]+>', '', text)


# Создаём глобальный обработчик
web_handler = WebSearchHandler()


# Функция для обратной совместимости
async def process_web_command(message: types.Message):
    """Обратная совместимость со старым API"""
    await web_handler.process_web_command(message)


# Дополнительные команды для расширенного функционала
async def process_web_help(message: types.Message):
    """Команда /webhelp - подробная справка"""
    help_text = (
        "📚 <b>Подробная справка по веб-поиску</b>\n\n"

        "<b>Основные возможности:</b>\n"
        "• Поиск актуальной информации в интернете\n"
        "• Автоматическое форматирование результатов\n"
        "• Поддержка ссылок и разметки\n"
        "• Защита от спама (rate limiting)\n\n"

        "<b>Советы для лучших результатов:</b>\n"
        "1. Формулируйте запросы конкретно\n"
        "2. Указывайте контекст (город, дата, тема)\n"
        "3. Используйте ключевые слова\n\n"

        "<b>Примеры хороших запросов:</b>\n"
        "✅ /web рестораны с веганским меню в центре Москвы\n"
        "✅ /web сравнение iPhone 15 и Samsung S24\n"
        "✅ /web курс доллара на сегодня\n\n"

        "<b>Ограничения:</b>\n"
        f"• Максимум {RATE_LIMIT_MAX_REQUESTS} запросов в минуту\n"
        "• Результаты могут быть разбиты на несколько сообщений\n\n"

        "<i>Совет: сохраняйте важные результаты, так как история поиска не сохраняется</i>"
    )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔍 Попробовать поиск", switch_inline_query_current_chat="/web ")]
    ])

    await message.answer(help_text, parse_mode="HTML", reply_markup=keyboard)


async def process_web_stats(message: types.Message):
    """Команда /webstats - статистика использования (для админов)"""
    # Здесь можно добавить проверку прав администратора
    user_id = message.from_user.id

    stats_text = (
        "📊 <b>Статистика веб-поиска</b>\n\n"
        f"👤 Ваш ID: <code>{user_id}</code>\n"
        f"🔍 Модель поиска: <code>{OPENAI_MODEL_WEBSEARCH}</code>\n"
        f"⏱ Лимит запросов: {RATE_LIMIT_MAX_REQUESTS} в {RATE_LIMIT_WINDOW} сек\n"
    )

    await message.answer(stats_text, parse_mode="HTML")
