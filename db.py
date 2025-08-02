# db.py
import asyncpg
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from asyncpg import Pool

from config import DATABASE_URL

pool: Pool = None


async def init_db():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

    conn = await pool.acquire()
    try:
        # Создаём таблицу embeddings (базовая структура)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id serial PRIMARY KEY,
                group_id bigint NOT NULL,
                user_id bigint NOT NULL,
                embedding_vector float8[] NOT NULL,
                created_at timestamptz DEFAULT now()
            );
        """)

        # Добавляем новые колонки к embeddings если их нет
        try:
            await conn.execute("ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS message_id bigint;")
        except:
            pass
        try:
            await conn.execute("ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS magnitude float8;")
        except:
            pass

        # Создаём индексы для embeddings
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_group_time 
                ON embeddings(group_id, created_at);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_user 
                ON embeddings(user_id);
        """)

        # Таблица messages (базовая структура)
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

        # Добавляем новые колонки к messages если их нет
        try:
            await conn.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS message_type text DEFAULT 'text';")
        except:
            pass
        try:
            await conn.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS reply_to_message_id bigint;")
        except:
            pass

        # Создаём индексы для messages
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_group_time 
                ON messages(group_id, created_at);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user 
                ON messages(user_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_reply 
                ON messages(reply_to_message_id);
        """)

        # Таблица для кэширования анализа
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id serial PRIMARY KEY,
                group_id bigint NOT NULL,
                analysis_type text NOT NULL,
                analysis_date date NOT NULL,
                result jsonb NOT NULL,
                created_at timestamptz DEFAULT now(),

                CONSTRAINT analysis_cache_unique 
                    UNIQUE (group_id, analysis_type, analysis_date)
            );
        """)

        # Таблица для хранения пользовательских метрик
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_metrics (
                id serial PRIMARY KEY,
                group_id bigint NOT NULL,
                user_id bigint NOT NULL,
                metric_date date NOT NULL,
                message_count int DEFAULT 0,
                avg_response_time float8,
                influence_score float8,
                sentiment_score float8,
                created_at timestamptz DEFAULT now(),

                CONSTRAINT user_metrics_unique 
                    UNIQUE (group_id, user_id, metric_date)
            );
        """)

    finally:
        await pool.release(conn)


async def save_embedding(group_id: int, user_id: int,
                         embedding_vector: list[float],
                         message_id: Optional[int] = None):
    """Сохранение эмбеддинга с вычислением magnitude"""
    magnitude = sum(x ** 2 for x in embedding_vector) ** 0.5

    conn = await pool.acquire()
    try:
        # Просто вставляем новую запись без ON CONFLICT
        await conn.execute(
            """
            INSERT INTO embeddings 
                (group_id, user_id, message_id, embedding_vector, magnitude)
            VALUES ($1, $2, $3, $4, $5)
        """, group_id, user_id, message_id, embedding_vector, magnitude)
    finally:
        await pool.release(conn)


async def count_embeddings() -> int:
    conn = await pool.acquire()
    try:
        row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM embeddings;")
        return row["cnt"]
    finally:
        await pool.release(conn)


async def save_message(group_id: int, user_id: int, user_name: str, text: str,
                       message_type: str = "text",
                       reply_to_message_id: Optional[int] = None):
    """Сохранение сообщения с расширенной информацией"""
    conn = await pool.acquire()
    try:
        result = await conn.fetchrow(
            """
            INSERT INTO messages 
                (group_id, user_id, user_name, text, message_type, reply_to_message_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """, group_id, user_id, user_name, text, message_type, reply_to_message_id)
        return result['id']
    finally:
        await pool.release(conn)


async def get_messages_for_period(group_id: int, start_time: datetime,
                                  end_time: datetime) -> List[Dict]:
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            SELECT 
                id, user_id, user_name, text, message_type, 
                reply_to_message_id, created_at
            FROM messages
            WHERE group_id = $1
              AND created_at >= $2
              AND created_at < $3
            ORDER BY created_at
        """, group_id, start_time, end_time)
        return [dict(row) for row in rows]
    finally:
        await pool.release(conn)


async def get_embeddings_for_period(group_id: int, start_time: datetime,
                                    end_time: datetime) -> List[Dict]:
    """Получение эмбеддингов за период"""
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            SELECT 
                e.id, e.user_id, e.message_id, e.embedding_vector, 
                e.magnitude, e.created_at, m.text, m.user_name
            FROM embeddings e
            LEFT JOIN messages m ON e.message_id = m.id
            WHERE e.group_id = $1
              AND e.created_at >= $2
              AND e.created_at < $3
            ORDER BY e.created_at
        """, group_id, start_time, end_time)
        return [dict(row) for row in rows]
    finally:
        await pool.release(conn)


async def get_user_statistics(group_id: int, start_time: datetime,
                              end_time: datetime) -> Dict[int, Dict]:
    """Получение статистики по пользователям"""
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            WITH user_messages AS (
                SELECT 
                    user_id,
                    user_name,
                    COUNT(*) as message_count,
                    AVG(LENGTH(text)) as avg_message_length,
                    COUNT(DISTINCT DATE_TRUNC('hour', created_at)) as active_hours
                FROM messages
                WHERE group_id = $1
                  AND created_at >= $2
                  AND created_at < $3
                GROUP BY user_id, user_name
            ),
            user_embeddings AS (
                SELECT 
                    user_id,
                    COUNT(*) as embedding_count,
                    AVG(magnitude) as avg_magnitude
                FROM embeddings
                WHERE group_id = $1
                  AND created_at >= $2
                  AND created_at < $3
                GROUP BY user_id
            )
            SELECT 
                um.*,
                ue.embedding_count,
                ue.avg_magnitude
            FROM user_messages um
            LEFT JOIN user_embeddings ue ON um.user_id = ue.user_id
            ORDER BY um.message_count DESC
        """, group_id, start_time, end_time)

        return {
            row['user_id']: {
                'user_name': row['user_name'],
                'message_count': row['message_count'],
                'avg_message_length': float(row['avg_message_length']),
                'active_hours': row['active_hours'],
                'embedding_count': row['embedding_count'] or 0,
                'avg_magnitude': float(row['avg_magnitude']) if row['avg_magnitude'] else 0
            }
            for row in rows
        }
    finally:
        await pool.release(conn)


async def get_group_statistics(group_id: int, start_time: datetime,
                               end_time: datetime) -> Dict[str, Any]:
    """Получение общей статистики группы"""
    conn = await pool.acquire()
    try:
        # Основная статистика
        basic_stats = await conn.fetchrow(
            """
            SELECT 
                COUNT(DISTINCT user_id) as active_users,
                COUNT(*) as total_messages,
                COUNT(DISTINCT DATE_TRUNC('hour', created_at)) as active_hours,
                MIN(created_at) as first_message,
                MAX(created_at) as last_message
            FROM messages
            WHERE group_id = $1
              AND created_at >= $2
              AND created_at < $3
        """, group_id, start_time, end_time)

        # Почасовая активность
        hourly_activity = await conn.fetch(
            """
            SELECT 
                EXTRACT(HOUR FROM created_at) as hour,
                COUNT(*) as message_count
            FROM messages
            WHERE group_id = $1
              AND created_at >= $2
              AND created_at < $3
            GROUP BY EXTRACT(HOUR FROM created_at)
            ORDER BY message_count DESC
            LIMIT 3
        """, group_id, start_time, end_time)

        peak_hours = [f"{int(row['hour'])}:00" for row in hourly_activity]

        # Статистика по типам сообщений
        message_types = await conn.fetch(
            """
            SELECT 
                message_type,
                COUNT(*) as count
            FROM messages
            WHERE group_id = $1
              AND created_at >= $2
              AND created_at < $3
            GROUP BY message_type
        """, group_id, start_time, end_time)

        return {
            'active_users': basic_stats['active_users'],
            'total_messages': basic_stats['total_messages'],
            'active_hours': basic_stats['active_hours'],
            'first_message': basic_stats['first_message'],
            'last_message': basic_stats['last_message'],
            'peak_hours': peak_hours,
            'message_types': {row['message_type']: row['count'] for row in message_types}
        }
    finally:
        await pool.release(conn)


async def get_message_clusters(group_id: int, start_time: datetime,
                               end_time: datetime, n_clusters: int = 5) -> List[Dict]:
    """Получение кластеров сообщений для анализа тем"""
    conn = await pool.acquire()
    try:
        # Эта функция будет использоваться для предварительной группировки
        # Реальная кластеризация происходит в Python
        rows = await conn.fetch(
            """
            SELECT 
                e.embedding_vector,
                e.user_id,
                e.created_at,
                m.text,
                m.user_name
            FROM embeddings e
            JOIN messages m ON e.message_id = m.id
            WHERE e.group_id = $1
              AND e.created_at >= $2
              AND e.created_at < $3
            ORDER BY e.created_at
        """, group_id, start_time, end_time)

        return [dict(row) for row in rows]
    finally:
        await pool.release(conn)


async def save_analysis_cache(group_id: int, analysis_type: str,
                              analysis_date: datetime, result: Dict):
    """Сохранение результатов анализа в кэш"""
    conn = await pool.acquire()
    try:
        import json
        await conn.execute(
            """
            INSERT INTO analysis_cache 
                (group_id, analysis_type, analysis_date, result)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (group_id, analysis_type, analysis_date)
            DO UPDATE SET 
                result = EXCLUDED.result,
                created_at = now()
        """, group_id, analysis_type, analysis_date.date(), json.dumps(result))
    finally:
        await pool.release(conn)


async def get_analysis_cache(group_id: int, analysis_type: str,
                             analysis_date: datetime) -> Optional[Dict]:
    """Получение закэшированного анализа"""
    conn = await pool.acquire()
    try:
        row = await conn.fetchrow(
            """
            SELECT result
            FROM analysis_cache
            WHERE group_id = $1
              AND analysis_type = $2
              AND analysis_date = $3
              AND created_at > $4
        """, group_id, analysis_type, analysis_date.date(),
            datetime.utcnow() - timedelta(hours=24))

        return dict(row['result']) if row else None
    finally:
        await pool.release(conn)


async def update_user_metrics(group_id: int, user_id: int,
                              metric_date: datetime, metrics: Dict):
    """Обновление метрик пользователя"""
    conn = await pool.acquire()
    try:
        await conn.execute(
            """
            INSERT INTO user_metrics 
                (group_id, user_id, metric_date, message_count, 
                 avg_response_time, influence_score, sentiment_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (group_id, user_id, metric_date)
            DO UPDATE SET 
                message_count = EXCLUDED.message_count,
                avg_response_time = EXCLUDED.avg_response_time,
                influence_score = EXCLUDED.influence_score,
                sentiment_score = EXCLUDED.sentiment_score,
                created_at = now()
        """, group_id, user_id, metric_date.date(),
            metrics.get('message_count', 0),
            metrics.get('avg_response_time'),
            metrics.get('influence_score'),
            metrics.get('sentiment_score'))
    finally:
        await pool.release(conn)


async def get_user_metrics_history(group_id: int, user_id: int,
                                   days: int = 30) -> List[Dict]:
    """Получение истории метрик пользователя"""
    conn = await pool.acquire()
    try:
        rows = await conn.fetch(
            """
            SELECT 
                metric_date,
                message_count,
                avg_response_time,
                influence_score,
                sentiment_score
            FROM user_metrics
            WHERE group_id = $1
              AND user_id = $2
              AND metric_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY metric_date DESC
        """ % days, group_id, user_id)

        return [dict(row) for row in rows]
    finally:
        await pool.release(conn)


async def find_similar_messages(group_id: int, embedding: List[float],
                                threshold: float = 0.8, limit: int = 10) -> List[Dict]:
    """Поиск похожих сообщений по эмбеддингу"""
    conn = await pool.acquire()
    try:
        # PostgreSQL pgvector extension would be ideal here
        # For now, we'll fetch all and calculate in Python
        rows = await conn.fetch(
            """
            SELECT 
                e.id,
                e.user_id,
                e.embedding_vector,
                e.created_at,
                m.text,
                m.user_name
            FROM embeddings e
            JOIN messages m ON e.message_id = m.id
            WHERE e.group_id = $1
            ORDER BY e.created_at DESC
            LIMIT 1000
        """, group_id)

        # Calculate similarities in Python
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        target_vec = np.array(embedding).reshape(1, -1)
        results = []

        for row in rows:
            vec = np.array(row['embedding_vector']).reshape(1, -1)
            similarity = cosine_similarity(target_vec, vec)[0][0]

            if similarity >= threshold:
                results.append({
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'user_name': row['user_name'],
                    'text': row['text'],
                    'created_at': row['created_at'],
                    'similarity': similarity
                })

        # Sort by similarity and limit
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    finally:
        await pool.release(conn)


async def get_conversation_threads(group_id: int, start_time: datetime,
                                   end_time: datetime) -> List[List[Dict]]:
    """Получение цепочек диалогов"""
    conn = await pool.acquire()
    try:
        # Получаем все сообщения с reply_to
        rows = await conn.fetch(
            """
            WITH RECURSIVE thread AS (
                -- Начальные сообщения (не ответы)
                SELECT 
                    id, user_id, user_name, text, 
                    reply_to_message_id, created_at,
                    id as thread_id, 0 as depth
                FROM messages
                WHERE group_id = $1
                  AND created_at >= $2
                  AND created_at < $3
                  AND reply_to_message_id IS NULL

                UNION ALL

                -- Рекурсивно находим ответы
                SELECT 
                    m.id, m.user_id, m.user_name, m.text,
                    m.reply_to_message_id, m.created_at,
                    t.thread_id, t.depth + 1
                FROM messages m
                JOIN thread t ON m.reply_to_message_id = t.id
                WHERE m.group_id = $1
            )
            SELECT * FROM thread
            ORDER BY thread_id, depth
        """, group_id, start_time, end_time)

        # Группируем по thread_id
        threads = {}
        for row in rows:
            thread_id = row['thread_id']
            if thread_id not in threads:
                threads[thread_id] = []
            threads[thread_id].append(dict(row))

        # Возвращаем только треды с более чем одним сообщением
        return [thread for thread in threads.values() if len(thread) > 1]

    finally:
        await pool.release(conn)
