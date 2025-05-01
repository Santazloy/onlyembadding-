# config.py
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PGHOST = os.getenv("PGHOST", "")
PGPORT = os.getenv("PGPORT", "")
PGUSER = os.getenv("PGUSER", "")
PGDATABASE = os.getenv("PGDATABASE", "")
PGPASSWORD = os.getenv("PGPASSWORD", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Список ID админов, разделённый запятой
ADMIN_IDS = os.getenv("ADMIN_IDS", "")
if ADMIN_IDS:
    ADMIN_IDS = [int(x.strip()) for x in ADMIN_IDS.split(",")]
else:
    ADMIN_IDS = []

# ID групп, в которых бот собирает эмбеддинги
GROUP_IDS = [
    -1002168406968,
    -1002433229203,
    -1002406184936,
    -1002315659294,
    -1002342166200,
    -1002250158149,
    -1002468561827,   # группа с китайским агентом
    -1002301241555    # рабочая группа команды
]

# Доп. данные по участникам (если нужно, храните где-то ещё)
TEAM_MEMBERS = {
    7987253532: "Анна",
    7281089930: "Жека",
    7935161063: "Лера",
    1720928807: "Катя",
    7894353415: "Босс",
    # ... и т.д.
}