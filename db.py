# db.py
import asyncpg
from config import DATABASE_URL
from asyncpg import Pool

# Пул соединений
pool: Pool = None

async def init_db():
    """Создаёт таблицу embeddings, если она не существует, и инициализирует пул соединений."""
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

    conn = await pool.acquire()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id serial PRIMARY KEY,
                group_id bigint NOT NULL,
                user_id bigint NOT NULL,
                embedding_vector float8[] NOT NULL,
                created_at timestamptz DEFAULT now()
            );
        """)
    finally:
        await pool.release(conn)


async def save_embedding(group_id: int, user_id: int, embedding_vector: list[float]):
    """Сохраняет эмбеддинг в таблицу."""
    conn = await pool.acquire()
    try:
        await conn.execute(
            """
            INSERT INTO embeddings (group_id, user_id, embedding_vector)
            VALUES ($1, $2, $3)
            """,
            group_id, user_id, embedding_vector
        )
    finally:
        await pool.release(conn)


async def count_embeddings() -> int:
    """Возвращает общее число записей в таблице эмбеддингов."""
    conn = await pool.acquire()
    try:
        row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM embeddings;")
        return row["cnt"]
    finally:
        await pool.release(conn)