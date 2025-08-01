# openai_utils.py

import os
import asyncio
import re
import logging
import json
from typing import List, Dict, Optional, Any
import openai

# Попытка создать клиент через новый класс OpenAI
try:
    from openai import OpenAI

    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
except ImportError:
    # fallback на модуль openai
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    _client = openai

logger = logging.getLogger(__name__)


async def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """
    Универсальный эмбеддинг с выбором модели.
    Модели:
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """
    loop = asyncio.get_event_loop()

    def _call():
        if hasattr(_client, "embeddings"):
            return _client.embeddings.create(model=model, input=text)
        else:
            return _client.Embedding.create(model=model, input=text)

    try:
        resp = await loop.run_in_executor(None, _call)
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"[OpenAI] get_embedding error: {e}")
        return []


async def get_embeddings_batch(texts: List[str],
                               model: str = "text-embedding-3-large") -> List[List[float]]:
    """
    Получение эмбеддингов для батча текстов (до 2048 текстов за раз)
    """
    loop = asyncio.get_event_loop()

    def _call():
        if hasattr(_client, "embeddings"):
            return _client.embeddings.create(model=model, input=texts)
        else:
            return _client.Embedding.create(model=model, input=texts)

    try:
        resp = await loop.run_in_executor(None, _call)
        return [item.embedding for item in resp.data]
    except Exception as e:
        logger.error(f"[OpenAI] get_embeddings_batch error: {e}")
        return [[] for _ in texts]


async def transcribe_audio(file_path: str) -> str:
    """
    Транскрибирует голосовое. Сначала пробует gpt-4o-transcribe,
    при ошибке откатывается на whisper-1.
    """
    loop = asyncio.get_event_loop()

    def _transcribe(model: str):
        if hasattr(_client, "audio"):  # новый SDK
            return _client.audio.transcriptions.create(
                model=model,
                file=open(file_path, "rb"),
            )
        # старый SDK
        return _client.Audio.transcribe(model, open(file_path, "rb"))

    try:
        try:
            resp = await loop.run_in_executor(None, _transcribe, "gpt-4o-transcribe")
        except Exception as primary_err:
            logging.warning("gpt-4o-transcribe failed → fallback: %s", primary_err)
            resp = await loop.run_in_executor(None, _transcribe, "whisper-1")

        # унификация доступа к тексту
        return resp.text if hasattr(resp, "text") else resp.get("text", "")
    except Exception as e:
        logging.error("[OpenAI] transcribe_audio error: %s", e)
        return ""


async def generate_text(
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
) -> str:
    """
    Универсальная генерация текста через ChatCompletion.
    """
    loop = asyncio.get_event_loop()

    def _call():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Если модель для web-search-preview, убираем параметры temperature/top_p
        if "search-preview" in model:
            if hasattr(_client, "chat"):
                return _client.chat.completions.create(
                    model=model,
                    messages=messages
                )
            else:
                return _client.ChatCompletion.create(
                    model=model,
                    messages=messages
                )
        # Иначе — полный набор параметров
        if hasattr(_client, "chat"):
            return _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0
            )
        else:
            return _client.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0
            )

    try:
        resp = await loop.run_in_executor(None, _call)
        if hasattr(_client, "chat"):
            return resp.choices[0].message.content.strip()
        else:
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"[OpenAI] generate_text error: {e}")
        return "Ошибка при генерации ответа."


async def generate_analysis_text(
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 4000
) -> str:
    """
    Генерация развёрнутого анализа (например, для daily_report).
    """
    system_prompt = """You are a professional business analyst specializing in team communication 
    and productivity analysis. Provide detailed, actionable insights based on data."""

    return await generate_text(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        system_prompt=system_prompt
    )


async def analyze_sentiment(text: str) -> float:
    """
    Анализ тональности текста.
    Возвращает значение от -1 (негативный) до 1 (позитивный)
    """
    prompt = f"""Analyze the sentiment of the following text and return ONLY a number between -1 and 1:
    -1 = very negative
    -0.5 = negative
    0 = neutral
    0.5 = positive
    1 = very positive

    Text: {text}

    Return only the number, nothing else."""

    try:
        result = await generate_text(
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0
        )
        return float(result.strip())
    except:
        return 0.0


async def extract_topics(texts: List[str], max_topics: int = 10) -> List[Dict[str, Any]]:
    """
    Извлечение основных тем из списка текстов
    """
    combined_text = "\n---\n".join(texts[:50])  # Ограничиваем количество

    prompt = f"""Extract the main topics from these messages. 
    Return a JSON array with up to {max_topics} topics.
    Each topic should have: {{"topic": "topic name", "relevance": 0.0-1.0, "keywords": ["word1", "word2"]}}

    Messages:
    {combined_text}

    Return only valid JSON array:"""

    try:
        result = await generate_text(
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.3
        )

        # Пытаемся распарсить JSON
        topics = json.loads(result.strip())
        return topics
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return []


async def summarize_conversation(messages: List[Dict[str, Any]],
                                 focus: Optional[str] = None) -> str:
    """
    Создание краткого саммари разговора
    """
    # Форматируем сообщения
    formatted_messages = []
    for msg in messages[:100]:  # Ограничиваем количество
        formatted_messages.append(f"{msg.get('user_name', 'Unknown')}: {msg.get('text', '')}")

    conversation = "\n".join(formatted_messages)

    focus_instruction = f"Focus particularly on: {focus}" if focus else ""

    prompt = f"""Create a concise summary of this conversation.
    {focus_instruction}

    Conversation:
    {conversation}

    Summary (max 200 words):"""

    return await generate_text(
        prompt=prompt,
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.5
    )


async def identify_action_items(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Извлечение action items из переписки
    """
    # Форматируем сообщения
    formatted_messages = []
    for msg in messages[:50]:
        formatted_messages.append(
            f"{msg.get('user_name', 'Unknown')} ({msg.get('created_at', '')}): {msg.get('text', '')}"
        )

    conversation = "\n".join(formatted_messages)

    prompt = f"""Extract action items from this conversation.
    Return a JSON array where each item has:
    {{"action": "what needs to be done", "assignee": "who should do it", "deadline": "when (if mentioned)"}}

    Conversation:
    {conversation}

    Return only valid JSON array:"""

    try:
        result = await generate_text(
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.2
        )

        action_items = json.loads(result.strip())
        return action_items
    except Exception as e:
        logger.error(f"Error extracting action items: {e}")
        return []


async def analyze_user_communication_style(user_messages: List[str]) -> Dict[str, Any]:
    """
    Анализ стиля коммуникации пользователя
    """
    if len(user_messages) < 5:
        return {"error": "Недостаточно сообщений для анализа"}

    sample_messages = "\n".join(user_messages[:20])

    prompt = f"""Analyze the communication style of this user based on their messages.
    Return a JSON object with:
    {{
        "tone": "formal/informal/mixed",
        "clarity": 0.0-1.0,
        "assertiveness": 0.0-1.0,
        "empathy": 0.0-1.0,
        "technical_level": "low/medium/high",
        "typical_message_length": "short/medium/long",
        "key_traits": ["trait1", "trait2", "trait3"]
    }}

    Messages:
    {sample_messages}

    Return only valid JSON:"""

    try:
        result = await generate_text(
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.3
        )

        analysis = json.loads(result.strip())
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing communication style: {e}")
        return {}


async def detect_emotions(text: str) -> Dict[str, float]:
    """
    Детектирование эмоций в тексте
    """
    prompt = f"""Analyze emotions in this text. Return JSON with scores (0.0-1.0) for:
    {{"joy": 0.0, "anger": 0.0, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0}}

    Text: {text}

    Return only valid JSON:"""

    try:
        result = await generate_text(
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0
        )

        emotions = json.loads(result.strip())
        return emotions
    except:
        return {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0, "disgust": 0}


async def generate_team_insights(team_data: Dict[str, Any]) -> str:
    """
    Генерация инсайтов о команде на основе данных
    """
    prompt = f"""Based on this team communication data, provide strategic insights:

Team Data:
- Active members: {team_data.get('active_users', 0)}
- Total messages: {team_data.get('total_messages', 0)}
- Average response time: {team_data.get('avg_response_time', 'N/A')}
- Topic clusters: {team_data.get('topic_clusters', [])}
- Sentiment distribution: {team_data.get('sentiment_dist', {})}

Provide insights on:
1. Team dynamics and collaboration patterns
2. Communication effectiveness
3. Potential areas of improvement
4. Recommended interventions

Keep insights specific and actionable."""

    return await generate_text(
        prompt=prompt,
        model="gpt-4o",
        max_tokens=1000,
        temperature=0.6
    )


def markdown_links_to_html(text: str) -> str:
    """
    [текст](url) -> <a href="url">текст</a>
    """
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    return re.sub(pattern, r'<a href="\2">\1</a>', text)


# Вспомогательные функции для работы с эмбеддингами

async def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Вычисление семантической схожести двух текстов
    """
    embeddings = await get_embeddings_batch([text1, text2])
    if not embeddings[0] or not embeddings[1]:
        return 0.0

    # Cosine similarity
    import numpy as np
    vec1 = np.array(embeddings[0])
    vec2 = np.array(embeddings[1])

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


async def cluster_texts_by_embedding(texts: List[str], n_clusters: int = 5) -> Dict[int, List[int]]:
    """
    Кластеризация текстов по эмбеддингам
    """
    if len(texts) < n_clusters:
        return {0: list(range(len(texts)))}

    # Получаем эмбеддинги
    embeddings = await get_embeddings_batch(texts)
    if not embeddings or not embeddings[0]:
        return {0: list(range(len(texts)))}

    # Кластеризация
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Группируем индексы по кластерам
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    return clusters
