import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Telegram Bot Token
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# OpenAI API Key
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# PostgreSQL (Supabase) параметры (необязательно, если используете DATABASE_URL)
PGHOST: str = os.getenv("PGHOST", "")
PGPORT: str = os.getenv("PGPORT", "")
PGUSER: str = os.getenv("PGUSER", "")
PGDATABASE: str = os.getenv("PGDATABASE", "")
PGPASSWORD: str = os.getenv("PGPASSWORD", "")

# Полный URL к базе (с SSL, если нужно)
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# Администраторы бота (по ID), передаются через ADMIN_IDS в .env
_admin_ids_raw = os.getenv("ADMIN_IDS", "")
if _admin_ids_raw:
    ADMIN_IDS = [int(x.strip()) for x in _admin_ids_raw.split(",") if x.strip()]
else:
    ADMIN_IDS = []

# Статический список ID групп, в которых бот собирает эмбеддинги и реагирует на команды
GROUP_IDS = [
    -1002168406968,
    -1002433229203,
    -1002406184936,
    -1002315659294,
    -1002342166200,
    -1002250158149,
    -1002468561827,  # группа с китайским агентом
    -1002301241555   # рабочая группа команды
]

# Дополнительные данные по участникам (если нужно)
TEAM_MEMBERS = {
    7987253532: "Royal",
    7281089930: "MIsha",
    7935161063: "Kris",
    1720928807: "Elizabeth",
    7894353415: "Leo",
    # ... и т.д.
}
