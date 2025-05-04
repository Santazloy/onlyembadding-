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

    Это позволит Telegram корректно парсить ссылки, 
    но если GPT выдаст прочие теги, 
    может вылезти ошибка.
    """
    # 1) временно вырезаем <a ...>...</a>
    #    (с учётом пробелов, кавычек и т.п.)
    import html
    placeholders = {}
    a_tags = re.findall(r"(<a\s+href=\".*?\">.*?</a>)", text, flags=re.DOTALL)
    for i, tag in enumerate(a_tags):
        placeholder = f"__A_TAG_{i}__"
        placeholders[placeholder] = tag
        text = text.replace(tag, placeholder, 1)

    # 2) экранируем <, >, &
    text = (text.replace("&", "&amp;").replace("<",
                                               "&lt;").replace(">", "&gt;"))

    # 3) возвращаем <a ...>...</a>
    for placeholder, tag in placeholders.items():
        text = text.replace(placeholder, tag, 1)

    return text


async def process_web_command(message: types.Message):
    """
    /web <запрос> – отправляем запрос к специальной модели (web-search-preview),
    результат подставляем в Telegram как HTML, 
    с заменой Markdown-ссылок -> <a>.
    """
    text = message.text.strip()
    if not text.lower().startswith("/web"):
        await message.answer("Использование: /web <запрос>")
        return

    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Укажите запрос после /web.")
        return

    query = parts[1].strip()
    if not query:
        await message.answer("Нужно указать непустой запрос.")
        return

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL_WEBSEARCH,
            messages=[{
                "role": "user",
                "content": query
            }],
            # temperature=0 для более фактического ответа
        )
    except Exception as e:
        await message.answer(f"Ошибка при запросе к веб-поиску: {e}")
        return

    raw_answer = response["choices"][0]["message"]["content"]

    # 1) Меняем Markdown-ссылки на <a>
    replaced = markdown_links_to_html(raw_answer)
    # 2) Минимально «очищаем» (чтобы Telegram не падал)
    final_html = sanitize_minimal(replaced)

    # Пытаемся отправить parse_mode=HTML
    try:
        await message.answer(
            final_html,
            parse_mode="HTML",
            disable_web_page_preview=
            False  # Или True, если хотите отключить превью
        )
    except Exception as e:
        # При ошибке отдадим «сырой» текст
        await message.answer(
            f"Ошибка при отправке HTML:\n{e}\n\nСырой ответ:\n{raw_answer}",
            disable_web_page_preview=True)
