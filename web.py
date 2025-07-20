import os
import re
import openai
from aiogram import types

OPENAI_MODEL_WEBSEARCH = "gpt-4o-search-preview"  # или вашу модель
openai.api_key = os.getenv("OPENAI_API_KEY", "")


def markdown_links_to_html(text: str) -> str:
    """
    Ищем в тексте Markdown-ссылки: [текст](ссылка)
    и конвертируем в <a href="ссылка">текст</a>.
    """
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    repl = r'<a href="\2">\1</a>'
    return re.sub(pattern, repl, text)


def sanitize_minimal(text: str) -> str:
    """
    Минимальная «санитизация»:
      - Заменим <, > и & на HTML-сущности,
      - но НЕ трогаем теги <a ...>...</a>.
    """
    placeholders = {}
    a_tags = re.findall(r"(<a\s+href=\".*?\">.*?</a>)", text, flags=re.DOTALL)
    for i, tag in enumerate(a_tags):
        ph = f"__A_TAG_{i}__"
        placeholders[ph] = tag
        text = text.replace(tag, ph, 1)

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for ph, tag in placeholders.items():
        text = text.replace(ph, tag, 1)

    return text


async def process_web_command(message: types.Message):
    """
    /web <запрос> – отправляем запрос к web-search-preview,
    результат подставляем в Telegram как HTML.
    """
    text = message.text.strip()
    if not text.lower().startswith("/web"):
        await message.answer("Использование: /web <запрос>")
        return

    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer("Укажите непустой запрос после /web.")
        return

    query = parts[1].strip()

    try:
        # Пробуем старый интерфейс (0.28)
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL_WEBSEARCH,
            messages=[{"role": "user", "content": query}],
            temperature=0
        )
        raw_answer = response["choices"][0]["message"]["content"]
    except (AttributeError, openai.error.InvalidRequestError):
        # Если не получилось — пробуем новый интерфейс (>=1.0.0)
        response = openai.chat.completions.create(
            model=OPENAI_MODEL_WEBSEARCH,
            messages=[{"role": "user", "content": query}],
            temperature=0
        )
        raw_answer = response.choices[0].message.content
    except Exception as e:
        await message.answer(f"Ошибка при запросе к веб-поиску: {e}")
        return

    # 1) Меняем Markdown-ссылки на <a>
    replaced = markdown_links_to_html(raw_answer)
    # 2) Минимальная «очистка» для безопасной отправки
    final_html = sanitize_minimal(replaced)

    try:
        await message.answer(
            final_html,
            parse_mode="HTML",
            disable_web_page_preview=False
        )
    except Exception as e:
        await message.answer(
            f"Ошибка при отправке HTML:\n{e}\n\nСырой ответ:\n{raw_answer}",
            disable_web_page_preview=True
        )
