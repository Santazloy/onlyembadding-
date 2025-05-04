# daily_report.py

import logging
import asyncio
import textwrap
from datetime import datetime, timedelta
from aiogram import Bot

from config import GROUP_IDS
from db import get_messages_for_period
from openai_utils import generate_analysis_text

logger = logging.getLogger(__name__)


async def send_reports_for_all_groups(bot: Bot):
    """
    Вызывается планировщиком каждый день в 00:00 (или когда вам нужно),
    или вручную по /report для всех групп.
    """
    for group_id in GROUP_IDS:
        try:
            await send_daily_report(bot, group_id)
        except Exception as e:
            logger.exception(
                f"Ошибка при генерации отчёта для группы {group_id}: {e}")


async def send_daily_report(bot: Bot, group_id: int):
    """
    Формирует и отправляет отчёт по чат-переписке за последние сутки.
    """
    now_utc = datetime.utcnow()
    day_ago_utc = now_utc - timedelta(days=1)

    # 1) Забираем сообщения из таблицы messages (за сутки)
    messages = await get_messages_for_period(group_id, day_ago_utc, now_utc)
    if not messages:
        await bot.send_message(
            group_id, "За прошедший день в этом чате сообщений не было.")
        return

    # Собираем тексты
    chat_text = "\n".join(f"{row['user_name']}: {row['text']}"
                          for row in messages)

    # 2) Формируем «усиленный» промпт
    prompt = f"""
Ниже приведена переписка (за последние 24 часа) в рабочей группе.

Тебе нужно сделать **глубокий и конкретный анализ** по следующим разделам. 
Формат ответа: **заголовок** каждого раздела в теге <code>...</code>, а сам анализ в теге <pre>...</pre>.
**Прошу давать конкретные примеры**, (обезличенные, без разглашения личных данных), и **четкие рекомендации**:

1) <code>Анализ тональности и настроений</code>  
<pre>
- Определи общий настрой (позитив, нейтральность, напряжение...).  
- Укажи, где были конфликты или недопонимания, если это видно.  
- Приведи 2–3 короткие цитаты (или пересказы сообщений), иллюстрирующие тон переписки.
</pre>

2) <code>Ключевые темы и вовлечённость</code>  
<pre>
- Какие основные темы обсуждались?  
- Насколько активно участники вовлечены?  
- Кто проявляет лидерство или инициативу?
</pre>

3) <code>Проблемные точки и конфликты</code>  
<pre>
- Выяви, были ли спорные моменты, токсичные/грубые высказывания.  
- Как участники реагировали и что помогло/не помогло уладить ситуацию.
</pre>

4) <code>Эффективность коммуникации</code>  
<pre>
- Насколько чётко люди формулируют мысли?  
- Есть ли повторяющиеся вопросы, задержки в ответах, непонятые задачи?
</pre>

5) <code>Анализ навыков и индивидуальных потребностей</code>  
<pre>
- Какие сильные стороны отдельных участников видны в переписке?  
- У кого, похоже, пробелы в знаниях или навыках?  
- Как это отражается на работе команды?
</pre>

6) <code>Уровень стресса и "перегрузки"</code>  
<pre>
- Видно ли по сообщениям, что кто-то перегружен задачами?  
- Есть ли признаки усталости, выгорания, жалоб?
</pre>

7) <code>Индикаторы продуктивности</code>  
<pre>
- Появились ли результаты, отчёты о выполненных задачах, успехах?  
- Упоминаются ли сроки, дедлайны, их срывы/выполнение?
</pre>

8) <code>Контроль качества коммуникации</code>  
<pre>
- Насколько вежлив и конструктивен тон сообщений?  
- Есть ли мат, резкие высказывания, переход на личности?
</pre>

9) <code>Рекомендации</code>  (в теге <pre>...</pre>)
<pre>
- Приведи конкретные практические шаги, как улучшить атмосферу, 
  что можно поправить в менеджменте, общении, вовлечённости.  
- Дай больше конкретики (2–4 пункта по каждому замечанию).
</pre>

Текст переписки:
{chat_text}
"""

    # 3) Вызываем GPT
    report = await generate_analysis_text(prompt)
    if not report:
        await bot.send_message(group_id,
                               "Ошибка: не удалось получить отчёт от GPT.")
        return

    # 4) Отправляем результат — возможно, разбивая по 4000 символов
    await send_long_html(bot, group_id, report)


def fix_html_tags(text: str) -> str:
    """
    Простейшая функция, которая пытается сбалансировать <pre>...</pre>.
    """
    text = text.strip()
    opens = text.count("<pre>")
    closes = text.count("</pre>")
    if opens > closes:
        diff = opens - closes
        text += "</pre>" * diff
    elif closes > opens:
        diff = closes - opens
        for _ in range(diff):
            idx = text.rfind("</pre>")
            if idx >= 0:
                text = text[:idx] + text[idx + 6:]
    return text


async def send_long_html(bot: Bot, chat_id: int, text: str, chunk_size=4000):
    """
    Дробим длинный текст и отправляем parse_mode=HTML.
    """
    text = fix_html_tags(text.strip())
    chunks = textwrap.wrap(text, width=chunk_size, replace_whitespace=False)
    for chunk in chunks:
        chunk = fix_html_tags(chunk)
        await bot.send_message(chat_id, chunk, parse_mode="HTML")
