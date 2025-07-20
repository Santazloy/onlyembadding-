import re
from aiogram import types
from openai_utils import generate_text, markdown_links_to_html

# Настройки для OpenAI Web Search Preview
OPENAI_MODEL_WEBSEARCH = "gpt-4o-search-preview"  # или ваша модель

def sanitize_minimal(text: str) -> str:
    """
    Минимальная «санитизация»:
      - Заменяем <, > и & на HTML-сущности,
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
        raw_answer = await generate_text(
            prompt=query,
            model=OPENAI_MODEL_WEBSEARCH
        )
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
