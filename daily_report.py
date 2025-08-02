# daily_report.py

import logging
import asyncio
import textwrap
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from aiogram import Bot
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config import GROUP_IDS
from db import (
    get_messages_for_period,
    get_embeddings_for_period,
    get_user_statistics,
    get_group_statistics,
    get_message_clusters
)
from openai_utils import generate_analysis_text, get_embedding

logger = logging.getLogger(__name__)


class AdvancedAnalyzer:
    """Продвинутый анализатор с использованием эмбеддингов"""

    def __init__(self):
        self.similarity_threshold = 0.75
        self.cluster_count = 5

    async def analyze_semantic_clusters(self, embeddings: List[Dict]) -> Dict[str, Any]:
        """Кластеризация сообщений по семантической близости"""
        if len(embeddings) < self.cluster_count:
            return {"clusters": [], "error": "Недостаточно данных для кластеризации"}

        # Извлекаем векторы
        vectors = np.array([emb['embedding_vector'] for emb in embeddings])
        user_ids = [emb['user_id'] for emb in embeddings]

        # K-means кластеризация
        kmeans = KMeans(n_clusters=min(self.cluster_count, len(vectors)), random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Анализ кластеров
        cluster_analysis = []
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_users = [user_ids[idx] for idx in cluster_indices]

            # Центроид кластера
            centroid = kmeans.cluster_centers_[i]

            cluster_analysis.append({
                "cluster_id": i,
                "size": len(cluster_indices),
                "dominant_users": Counter(cluster_users).most_common(3),
                "centroid": centroid.tolist()[:10]  # Первые 10 измерений для краткости
            })

        return {"clusters": cluster_analysis, "total_messages": len(embeddings)}

    async def analyze_user_evolution(self, user_embeddings: Dict[int, List]) -> Dict[int, Dict]:
        """Анализ эволюции тем и стиля каждого пользователя"""
        evolution_analysis = {}

        for user_id, embeddings in user_embeddings.items():
            if len(embeddings) < 2:
                continue

            # Сортируем по времени
            embeddings.sort(key=lambda x: x['created_at'])

            # Считаем семантический дрейф
            semantic_drift = []
            for i in range(1, len(embeddings)):
                prev_vec = np.array(embeddings[i - 1]['embedding_vector'])
                curr_vec = np.array(embeddings[i]['embedding_vector'])
                similarity = cosine_similarity([prev_vec], [curr_vec])[0][0]
                semantic_drift.append(1 - similarity)

            # Анализ постоянства тем
            all_vectors = [np.array(e['embedding_vector']) for e in embeddings]
            avg_similarity = np.mean(cosine_similarity(all_vectors))

            evolution_analysis[user_id] = {
                "semantic_drift": np.mean(semantic_drift) if semantic_drift else 0,
                "topic_consistency": avg_similarity,
                "message_count": len(embeddings),
                "activity_span": (embeddings[-1]['created_at'] - embeddings[0]['created_at']).total_seconds() / 3600
            }

        return evolution_analysis

    async def detect_conversation_patterns(self, messages: List[Dict], embeddings: List[Dict]) -> Dict:
        """Определение паттернов разговора через эмбеддинги"""
        # Создаем временную последовательность
        timeline = []
        emb_dict = {(e['user_id'], e['created_at']): e['embedding_vector']
                    for e in embeddings}

        for msg in messages:
            key = (msg['user_id'], msg['created_at'])
            if key in emb_dict:
                timeline.append({
                    'time': msg['created_at'],
                    'user_id': msg['user_id'],
                    'text': msg['text'],
                    'embedding': emb_dict[key]
                })

        # Анализ диалогов (ответы в течение 5 минут)
        dialogues = []
        for i in range(1, len(timeline)):
            time_diff = (timeline[i]['time'] - timeline[i - 1]['time']).total_seconds()
            if time_diff < 300 and timeline[i]['user_id'] != timeline[i - 1]['user_id']:
                vec1 = np.array(timeline[i - 1]['embedding'])
                vec2 = np.array(timeline[i]['embedding'])
                similarity = cosine_similarity([vec1], [vec2])[0][0]

                dialogues.append({
                    'users': (timeline[i - 1]['user_id'], timeline[i]['user_id']),
                    'similarity': similarity,
                    'time_diff': time_diff
                })

        # Статистика диалогов
        dialogue_stats = {
            'total_dialogues': len(dialogues),
            'avg_response_time': np.mean([d['time_diff'] for d in dialogues]) if dialogues else 0,
            'avg_relevance': np.mean([d['similarity'] for d in dialogues]) if dialogues else 0,
            'high_relevance_count': sum(1 for d in dialogues if d['similarity'] > 0.8)
        }

        return dialogue_stats

    async def calculate_influence_score(self, user_stats: Dict, group_embeddings: List[Dict]) -> Dict[int, float]:
        """Расчет влияния каждого пользователя на дискуссию"""
        influence_scores = {}

        # Группируем эмбеддинги по пользователям
        user_embeddings = defaultdict(list)
        for emb in group_embeddings:
            user_embeddings[emb['user_id']].append(emb['embedding_vector'])

        # Для каждого пользователя
        for user_id, vectors in user_embeddings.items():
            if not vectors:
                continue

            # Средний вектор пользователя
            user_avg_vector = np.mean(vectors, axis=0)

            # Считаем, насколько последующие сообщения других похожи на сообщения этого пользователя
            influence = 0
            influence_count = 0

            for i, emb in enumerate(group_embeddings):
                if emb['user_id'] == user_id:
                    # Ищем ответы других пользователей в следующие 10 сообщений
                    for j in range(i + 1, min(i + 11, len(group_embeddings))):
                        if group_embeddings[j]['user_id'] != user_id:
                            other_vec = np.array(group_embeddings[j]['embedding_vector'])
                            similarity = cosine_similarity([user_avg_vector], [other_vec])[0][0]
                            influence += similarity
                            influence_count += 1

            influence_scores[user_id] = influence / influence_count if influence_count > 0 else 0

        return influence_scores


async def send_reports_for_all_groups(bot: Bot):
    """
    Вызывается планировщиком каждый день в 00:00 для автоматических отчетов
    """
    for group_id in GROUP_IDS:
        try:
            await send_daily_report(bot, group_id)
        except Exception as e:
            logger.exception(
                f"Ошибка при генерации отчёта для группы {group_id}: {e}")


async def send_daily_report(bot: Bot, group_id: int):
    """
    Отчёт за последние 24 часа (для обратной совместимости и автоматических отчетов)
    """
    await send_period_report(bot, group_id, days=1)


async def send_period_report(bot: Bot, group_id: int, days: int):
    """
    Универсальная функция для отчётов за разные периоды
    """
    now_utc = datetime.utcnow()
    start_time = now_utc - timedelta(days=days)

    # Определяем название периода для отчета
    period_names = {
        1: "24 часа",
        3: "72 часа",
        7: "неделю",
        30: "месяц"
    }
    period_name = period_names.get(days, f"{days} дней")

    analyzer = AdvancedAnalyzer()

    # 1) Получаем все данные
    messages = await get_messages_for_period(group_id, start_time, now_utc)
    if not messages:
        await bot.send_message(
            group_id, f"За последние {period_name} в этом чате сообщений не было.")
        return

    embeddings = await get_embeddings_for_period(group_id, start_time, now_utc)
    user_stats = await get_user_statistics(group_id, start_time, now_utc)
    group_stats = await get_group_statistics(group_id, start_time, now_utc)

    # 2) Продвинутый анализ
    cluster_analysis = await analyzer.analyze_semantic_clusters(embeddings)

    # Группируем эмбеддинги по пользователям
    user_embeddings = defaultdict(list)
    for emb in embeddings:
        user_embeddings[emb['user_id']].append(emb)

    evolution_analysis = await analyzer.analyze_user_evolution(user_embeddings)
    dialogue_patterns = await analyzer.detect_conversation_patterns(messages, embeddings)
    influence_scores = await analyzer.calculate_influence_score(user_stats, embeddings)

    # 3) Адаптируем количество сообщений для промпта в зависимости от периода
    message_limit = min(100 * days, 500)  # Больше сообщений для длинных периодов
    chat_text = "\n".join(f"{row['user_name']}: {row['text']}"
                          for row in messages[:message_limit])

    # 4) Специальные инструкции для разных периодов
    period_specific_instructions = {
        1: "Фокусируйся на оперативных задачах и ежедневной динамике.",
        3: "Обрати внимание на краткосрочные тренды и завершение недельных задач.",
        7: "Проанализируй недельные паттерны, прогресс по проектам и командную динамику.",
        30: "Сделай стратегический анализ: долгосрочные тренды, эволюция команды, достижение месячных целей."
    }

    specific_instruction = period_specific_instructions.get(days,
                                                            f"Проанализируй динамику за {days} дней, выдели ключевые изменения и тренды.")

    prompt = f"""
Ты профессиональный аналитик корпоративных коммуникаций. Проанализируй данные рабочего чата за последние {period_name}.

ПЕРИОД АНАЛИЗА: {period_name} (с {start_time.strftime('%d.%m.%Y')} по {now_utc.strftime('%d.%m.%Y')})
СПЕЦИАЛЬНАЯ ИНСТРУКЦИЯ: {specific_instruction}

ВХОДНЫЕ ДАННЫЕ:

1. СТАТИСТИКА ГРУППЫ:
- Всего сообщений: {group_stats.get('total_messages', 0)}
- Активных пользователей: {group_stats.get('active_users', 0)}
- Пиковые часы активности: {group_stats.get('peak_hours', 'н/д')}
- Среднее сообщений в день: {group_stats.get('total_messages', 0) / days:.1f}

2. КЛАСТЕРНЫЙ АНАЛИЗ (семантические группы тем):
{format_cluster_analysis(cluster_analysis)}

3. АНАЛИЗ ЭВОЛЮЦИИ ПОЛЬЗОВАТЕЛЕЙ:
{format_evolution_analysis(evolution_analysis, user_stats)}

4. ПАТТЕРНЫ ДИАЛОГОВ:
- Всего диалогов: {dialogue_patterns['total_dialogues']}
- Среднее время ответа: {dialogue_patterns['avg_response_time']:.1f} сек
- Средняя релевантность ответов: {dialogue_patterns['avg_relevance']:.2%}
- Высокорелевантных ответов: {dialogue_patterns['high_relevance_count']}

5. ИНДЕКС ВЛИЯНИЯ УЧАСТНИКОВ:
{format_influence_scores(influence_scores, user_stats)}

ФРАГМЕНТЫ ПЕРЕПИСКИ:
{chat_text}

ЗАДАНИЕ:
Создай профессиональный аналитический отчёт за {period_name} со следующей структурой. Используй теги <code> для заголовков и <pre> для содержания.

<code>📊 EXECUTIVE SUMMARY за {period_name}</code>
<pre>
Краткое резюме ключевых находок (3-5 пунктов):
- Главные темы периода
- Ключевые достижения/проблемы
- Критические риски
- Общая динамика команды
{f'- Сравнение с предыдущим периодом' if days > 1 else ''}
</pre>

<code>👥 ИНДИВИДУАЛЬНАЯ АНАЛИТИКА</code>
<pre>
Для каждого активного участника (топ-5 по активности):
[Имя участника]:
• Роль в команде: (лидер/исполнитель/эксперт/координатор)
• Вклад за период: (конкретные действия/решения)
• Сильные стороны: (что делает хорошо)
• Зоны развития: (что можно улучшить)
• Индекс влияния: X.XX (насколько влияет на дискуссию)
• Семантический дрейф: X.XX (стабильность тем)
{f'• Динамика за период: (рост/спад активности)' if days > 1 else ''}
</pre>

<code>🎯 АНАЛИЗ ПРОДУКТИВНОСТИ</code>
<pre>
• Выполненные задачи: (список с исполнителями)
• Незавершённые задачи: (с указанием блокеров)
• Эффективность коммуникаций: X/10
• Скорость принятия решений: (быстро/средне/медленно)
• Качество обратной связи: (примеры)
{f'• Прогресс по долгосрочным целям' if days >= 7 else ''}
</pre>

<code>🔍 СЕМАНТИЧЕСКИЙ АНАЛИЗ</code>
<pre>
• Основные тематические кластеры:
  1. [Тема] - XX% дискуссий
  2. [Тема] - XX% дискуссий
  3. [Тема] - XX% дискуссий
• Эмоциональный фон: (позитивный/нейтральный/напряжённый)
• Уровень вовлечённости: X/10
{f'• Эволюция тем за период' if days > 1 else ''}
</pre>

<code>⚠️ РИСКИ И ПРОБЛЕМНЫЕ ЗОНЫ</code>
<pre>
• Коммуникационные риски: (недопонимания, конфликты)
• Операционные риски: (срывы сроков, качество)
• Командные риски: (выгорание, демотивация)
• Признаки стресса у: [список участников]
{f'• Накопленные проблемы за период' if days >= 7 else ''}
</pre>

<code>📈 РЕКОМЕНДАЦИИ</code>
<pre>
{'Немедленные действия:' if days == 1 else f'Приоритетные действия на следующие {days} дней:'}
1. [Что сделать] - ответственный: [Кто]
2. [Что сделать] - ответственный: [Кто]
3. [Что сделать] - ответственный: [Кто]

{'Стратегические рекомендации:' if days >= 7 else 'Тактические улучшения:'}
• По улучшению процессов:
• По развитию команды:
• По коммуникациям:
</pre>

<code>📊 МЕТРИКИ ДЛЯ ОТСЛЕЖИВАНИЯ</code>
<pre>
• Количество решённых вопросов: X
• Среднее время на решение: X часов
• Индекс командного взаимодействия: X.X/10
• Прогресс по KPI: XX%
{f'• Тренд за период: ↑↓→' if days > 1 else ''}
</pre>

Используй данные из анализа эмбеддингов для обоснования выводов. Будь конкретен, приводи примеры.
Адаптируй глубину анализа под период: {period_name}.
"""

    # 4) Генерируем отчёт
    report = await generate_analysis_text(prompt, model="gpt-4o", max_tokens=4000)
    if not report:
        await bot.send_message(group_id,
                               "Ошибка: не удалось получить отчёт от GPT.")
        return

    # 5) Добавляем статистическое приложение
    stats_appendix = await format_statistical_appendix(user_stats, group_stats, influence_scores, days)
    full_report = report + "\n\n" + stats_appendix

    # 6) Отправляем результат
    await send_long_html(bot, group_id, full_report)


def format_cluster_analysis(cluster_analysis: Dict) -> str:
    """Форматирование результатов кластерного анализа"""
    if 'error' in cluster_analysis:
        return cluster_analysis['error']

    result = []
    for cluster in cluster_analysis.get('clusters', []):
        users_str = ", ".join([f"User_{uid} ({count})"
                               for uid, count in cluster['dominant_users']])
        result.append(f"Кластер {cluster['cluster_id']}: "
                      f"{cluster['size']} сообщений, "
                      f"основные участники: {users_str}")

    return "\n".join(result) or "Кластеризация не выполнена"


def format_evolution_analysis(evolution: Dict, user_stats: Dict) -> str:
    """Форматирование анализа эволюции пользователей"""
    result = []
    for user_id, analysis in evolution.items():
        user_name = user_stats.get(user_id, {}).get('user_name', f'User_{user_id}')
        result.append(
            f"{user_name}: "
            f"дрейф тем: {analysis['semantic_drift']:.2f}, "
            f"постоянство: {analysis['topic_consistency']:.2f}, "
            f"сообщений: {analysis['message_count']}"
        )

    return "\n".join(result[:10]) or "Нет данных"  # Топ-10


def format_influence_scores(scores: Dict, user_stats: Dict) -> str:
    """Форматирование индексов влияния"""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []

    for user_id, score in sorted_scores[:10]:  # Топ-10
        user_name = user_stats.get(user_id, {}).get('user_name', f'User_{user_id}')
        result.append(f"{user_name}: {score:.3f}")

    return "\n".join(result) or "Нет данных"


async def format_statistical_appendix(user_stats: Dict, group_stats: Dict,
                                      influence_scores: Dict, days: int = 1) -> str:
    """Форматирование статистического приложения"""
    period_name = {1: "24 часа", 3: "72 часа", 7: "неделю", 30: "месяц"}.get(days, f"{days} дней")

    appendix = f"<code>📈 СТАТИСТИЧЕСКОЕ ПРИЛОЖЕНИЕ за {period_name}</code>\n<pre>\n"

    # Топ участников по сообщениям
    appendix += "ТОП-10 ПО АКТИВНОСТИ:\n"
    sorted_users = sorted(user_stats.items(),
                          key=lambda x: x[1].get('message_count', 0),
                          reverse=True)[:10]

    for i, (user_id, stats) in enumerate(sorted_users, 1):
        msg_count = stats.get('message_count', 0)
        avg_per_day = msg_count / days
        appendix += (f"{i}. {stats.get('user_name', f'User_{user_id}')}: "
                     f"{msg_count} сообщений ({avg_per_day:.1f}/день), "
                     f"влияние: {influence_scores.get(user_id, 0):.3f}\n")

    # Временное распределение
    appendix += f"\nПИКОВЫЕ ЧАСЫ: {group_stats.get('peak_hours', 'н/д')}\n"
    appendix += f"ВСЕГО УНИКАЛЬНЫХ ТЕМ: {group_stats.get('unique_topics', 'н/д')}\n"

    # Дополнительная статистика для длинных периодов
    if days >= 7:
        appendix += f"\nСРЕДНЯЯ АКТИВНОСТЬ ПО ДНЯМ:\n"
        appendix += f"Рабочие дни: {group_stats.get('weekday_avg', 'н/д')} сообщений\n"
        appendix += f"Выходные: {group_stats.get('weekend_avg', 'н/д')} сообщений\n"

    if days >= 30:
        appendix += f"\nДИНАМИКА ПО НЕДЕЛЯМ:\n"
        appendix += f"Неделя 1: {group_stats.get('week1_messages', 'н/д')} сообщений\n"
        appendix += f"Неделя 2: {group_stats.get('week2_messages', 'н/д')} сообщений\n"
        appendix += f"Неделя 3: {group_stats.get('week3_messages', 'н/д')} сообщений\n"
        appendix += f"Неделя 4: {group_stats.get('week4_messages', 'н/д')} сообщений\n"

    appendix += "</pre>"
    return appendix


def fix_html_tags(text: str) -> str:
    """
    Простейшая функция, которая пытается сбалансировать <pre>...</pre>.
    """
    text = text.strip()
    opens = text.count("<pre>")
    closes = text.count("</pre>")
    if opens > closes:
        diff = opens - closes
        text += "</pre>" * diff
    elif closes > opens:
        diff = closes - opens
        for _ in range(diff):
            idx = text.rfind("</pre>")
            if idx >= 0:
                text = text[:idx] + text[idx + 6:]
    return text


async def send_long_html(bot: Bot, chat_id: int, text: str, chunk_size=4000):
    """
    Дробим длинный текст и отправляем parse_mode=HTML.
    """
    text = fix_html_tags(text.strip())

    # Если текст короткий, отправляем целиком
    if len(text) <= chunk_size:
        try:
            await bot.send_message(chat_id, text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Ошибка отправки HTML: {e}")
            # Fallback на обычный текст
            await bot.send_message(chat_id, text)
        return

    # Разбиваем длинный текст
    chunks = []
    current_chunk = ""
    lines = text.split('\n')

    for line in lines:
        # Если добавление строки превысит лимит, сохраняем текущий чанк
        if len(current_chunk) + len(line) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += '\n'
            current_chunk += line

    # Добавляем последний чанк
    if current_chunk:
        chunks.append(current_chunk)

    # Отправляем чанки
    for i, chunk in enumerate(chunks):
        chunk = fix_html_tags(chunk)
        try:
            await bot.send_message(chat_id, chunk, parse_mode="HTML")
            # Небольшая задержка между сообщениями
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Ошибка отправки чанка {i + 1}: {e}")
            # Пробуем отправить без HTML
            await bot.send_message(chat_id, chunk)
