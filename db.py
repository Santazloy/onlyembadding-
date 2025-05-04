# db.py
import asyncpg
from config import DATABASE_URL
from asyncpg import Pool

pool: Pool = None


async def init_db():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

    conn = await pool.acquire()
    try:
        # Создаём таблицу embeddings
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id serial PRIMARY KEY,
                group_id bigint NOT NULL,
                user_id bigint NOT NULL,
                embedding_vector float8[] NOT NULL,
                created_at timestamptz DEFAULT now()
            );
        """)

        # Таблица messages (для сохранения текста)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id serial PRIMARY KEY,
                group_id bigint NOT NULL,
                user_id bigint NOT NULL,
                user_name text,
                text text NOT NULL,
                created_at timestamptz DEFAULT now()
            );
        """)
    finally:
        await pool.release(conn)


async def save_embedding(group_id: int, user_id: int,
                         embedding_vector: list[float]):
    conn = await pool.acquire()
    try:
        await conn.execute(
            """
            INSERT INTO embeddings (group_id, user_id, embedding_vector)
            VALUES ($1, $2, $3)
        """, group_id, user_id, embedding_vector)
    finally:
        await pool.release(conn)


async def count_embeddings() -> int:
    conn = await pool.acquire()
    try:
        row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM embeddings;")
        return row["cnt"]
    finally:
        await pool.release(conn)


async def save_message(group_id: int, user_id: int, user_name: str, text: str):
    conn = await pool.acquire()
    try:
        await conn.execute(
            """
            INSERT INTO messages (group_id, user_id, user_name, text)
            VALUES ($1, $2, $3, $4)
        """, group_id, user_id, user_name, text)
    finally:
        await pool.release(conn)


async def get_messages_for_period(group_id: int, start_time, end_time):
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            SELECT user_name, text, created_at
            FROM messages
            WHERE group_id = $1
              AND created_at >= $2
              AND created_at < $3
            ORDER BY created_at
        """, group_id, start_time, end_time)
        return rows
    finally:
        await pool.release(conn)
