# example.py
from supabase import create_client, Client
import os

# Получаем URL и ключ из переменных окружения
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Создаем клиента Supabase
supabase: Client = create_client(url, key)

# Пример получения данных из таблицы
data = supabase.table("embeddings").select("*").execute()
print(data)