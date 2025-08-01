import os
import random
import logging
import asyncio
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openai_utils import generate_text

logger = logging.getLogger(__name__)

# Константы
TARO_FOLDER = Path(__file__).parent / "TARO"
DEFAULT_REVERSED_CHANCE = 0.2
DEFAULT_CARD_COUNT = 3
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 2000

# Для обратной совместимости
REVERSED_CHANCE = DEFAULT_REVERSED_CHANCE


class Suit(Enum):
    """Масти карт Таро"""
    WATER = "Воды"
    AIR = "Воздуха"
    EARTH = "Земли"
    FIRE = "Огня"
    MAJOR = "Старшие Арканы"


class CardType(Enum):
    """Типы карт"""
    MAJOR_ARCANA = "major"
    MINOR_ARCANA = "minor"


@dataclass
class TarotCard:
    """Представление карты Таро"""
    filename: str
    name: str
    suit: Suit
    card_type: CardType
    number: Optional[int] = None

    def __str__(self) -> str:
        return self.name


@dataclass
class DrawnCard:
    """Вытянутая карта с ориентацией"""
    card: TarotCard
    is_reversed: bool
    position: Optional[str] = None

    @property
    def orientation(self) -> str:
        return "перевёрнутая" if self.is_reversed else "прямая"

    @property
    def full_name(self) -> str:
        return f"{self.card.name} ({self.orientation})"


class TarotDeck:
    """Колода карт Таро Манара"""

    def __init__(self):
        self.cards: Dict[str, TarotCard] = {}
        self._initialize_deck()

    def _initialize_deck(self):
        """Инициализация колоды"""
        # Старшие арканы
        major_arcana = {
            "Шут.jpg": ("Шут", 0),
            "Маг.jpg": ("Маг", 1),
            "Верховная_жрица.jpg": ("Верховная Жрица", 2),
            "Императрица.jpg": ("Императрица", 3),
            "Император.jpg": ("Император", 4),
            "Жрец.jpg": ("Жрец (Иерофант)", 5),
            "Влюбленные.jpg": ("Влюблённые", 6),
            "Колесница.jpg": ("Колесница", 7),
            "Сила.jpg": ("Сила", 8),
            "Отшельник.jpg": ("Отшельник", 9),
            "Колесо_Фортуны.jpg": ("Колесо Фортуны", 10),
            "Правосудие.jpg": ("Правосудие", 11),
            "Наказание.jpg": ("Наказание (Повешенный)", 12),
            "Смерть.jpg": ("Смерть", 13),
            "Умеренность.jpg": ("Умеренность", 14),
            "Дьявол.jpg": ("Дьявол", 15),
            "Башня.jpg": ("Башня", 16),
            "Звезда.jpg": ("Звезда", 17),
            "Зеркало.jpg": ("Зеркало (Луна)", 18),
            "Солнце.jpg": ("Солнце", 19),
            "Суд.jpg": ("Суд (Страшный Суд)", 20),
            "Мир.jpg": ("Мир", 21)
        }

        for filename, (name, number) in major_arcana.items():
            self.cards[filename] = TarotCard(
                filename=filename,
                name=name,
                suit=Suit.MAJOR,
                card_type=CardType.MAJOR_ARCANA,
                number=number
            )

        # Младшие арканы
        minor_suits = {
            Suit.WATER: "воды",
            Suit.AIR: "воздуха",
            Suit.EARTH: "земли",
            Suit.FIRE: "огня"
        }

        court_cards = {
            "Слуга": "Паж (Слуга)",
            "Всадница": "Всадница (Рыцарь)",
            "Королева": "Королева",
            "Король": "Король"
        }

        for suit, suit_name in minor_suits.items():
            # Числовые карты
            self._add_card(f"Туз_{suit_name}.jpg", f"Туз {suit.value}", suit, 1)

            for i in range(2, 11):
                number_name = self._get_number_name(i)
                self._add_card(
                    f"{i}_{suit_name}.jpg",
                    f"{number_name} {suit.value}",
                    suit,
                    i
                )

            # Придворные карты
            for court_prefix, court_name in court_cards.items():
                self._add_card(
                    f"{court_prefix}_{suit_name}.jpg",
                    f"{court_name} {suit.value}",
                    suit
                )

    def _add_card(self, filename: str, name: str, suit: Suit,
                  number: Optional[int] = None):
        """Добавить карту в колоду"""
        self.cards[filename] = TarotCard(
            filename=filename,
            name=name,
            suit=suit,
            card_type=CardType.MINOR_ARCANA if suit != Suit.MAJOR else CardType.MAJOR_ARCANA,
            number=number
        )

    @staticmethod
    def _get_number_name(number: int) -> str:
        """Получить текстовое название числа"""
        names = {
            2: "Двойка", 3: "Тройка", 4: "Четвёрка", 5: "Пятёрка",
            6: "Шестёрка", 7: "Семёрка", 8: "Восьмёрка",
            9: "Девятка", 10: "Десятка"
        }
        return names.get(number, str(number))

    def draw_cards(self, count: int = DEFAULT_CARD_COUNT,
                   reversed_chance: float = DEFAULT_REVERSED_CHANCE,
                   positions: Optional[List[str]] = None) -> List[DrawnCard]:
        """
        Вытянуть карты из колоды

        Args:
            count: Количество карт
            reversed_chance: Вероятность перевёрнутой карты
            positions: Позиции для карт (опционально)

        Returns:
            Список вытянутых карт
        """
        if count > len(self.cards):
            raise ValueError(f"В колоде только {len(self.cards)} карт")

        if positions and len(positions) != count:
            raise ValueError("Количество позиций должно совпадать с количеством карт")

        selected_cards = random.sample(list(self.cards.values()), k=count)
        drawn_cards = []

        for i, card in enumerate(selected_cards):
            is_reversed = random.random() < reversed_chance
            position = positions[i] if positions else None
            drawn_cards.append(DrawnCard(
                card=card,
                is_reversed=is_reversed,
                position=position
            ))

        return drawn_cards

    def get_card_by_name(self, name: str) -> Optional[TarotCard]:
        """Найти карту по имени"""
        for card in self.cards.values():
            if card.name.lower() == name.lower():
                return card
        return None

    def get_cards_by_suit(self, suit: Suit) -> List[TarotCard]:
        """Получить все карты определённой масти"""
        return [card for card in self.cards.values() if card.suit == suit]


class TarotInterpreter:
    """Интерпретатор карт Таро"""

    def __init__(self, model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens

    async def interpret_card(self, drawn_card: DrawnCard,
                             context: Optional[str] = None) -> str:
        """
        Получить интерпретацию карты

        Args:
            drawn_card: Вытянутая карта
            context: Дополнительный контекст для интерпретации

        Returns:
            Интерпретация карты
        """
        system_prompt = (
            "Ты — профессиональный таролог, специализирующийся на колоде Таро Манара. "
            "Даёшь глубокие, психологически обоснованные интерпретации, "
            "учитывая эротическую символику колоды Манара. "
            "Отвечаешь дружелюбно и тактично, без излишней мистики."
        )

        position_text = f" в позиции '{drawn_card.position}'" if drawn_card.position else ""
        context_text = f"\n\nКонтекст вопроса: {context}" if context else ""

        user_prompt = (
            f"Карта: '{drawn_card.card.name}'{position_text}\n"
            f"Ориентация: {drawn_card.orientation}\n"
            f"Масть: {drawn_card.card.suit.value}"
            f"{context_text}\n\n"
            "Дай подробную интерпретацию этой карты, учитывая:\n"
            "1. Особенности колоды Манара и её символику\n"
            "2. Влияние ориентации карты\n"
            "3. Практические советы для вопрошающего"
        )

        try:
            interpretation = await generate_text(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                model=self.model,
                max_tokens=self.max_tokens
            )
            return interpretation.strip()
        except Exception as e:
            logger.error(f"Ошибка при интерпретации карты {drawn_card.card.name}: {e}")
            return self._get_fallback_interpretation(drawn_card)

    async def interpret_spread(self, drawn_cards: List[DrawnCard],
                               spread_type: str = "simple",
                               question: Optional[str] = None) -> str:
        """
        Интерпретировать расклад целиком

        Args:
            drawn_cards: Список вытянутых карт
            spread_type: Тип расклада
            question: Вопрос вопрошающего

        Returns:
            Общая интерпретация расклада
        """
        cards_description = "\n".join([
            f"{i + 1}. {card.full_name}"
            f"{f' - {card.position}' if card.position else ''}"
            for i, card in enumerate(drawn_cards)
        ])

        system_prompt = (
            "Ты — мастер Таро Манара. Даёшь целостную интерпретацию раскладов, "
            "видя связи между картами и их общее послание."
        )

        question_text = f"\nВопрос: {question}" if question else ""

        user_prompt = (
            f"Тип расклада: {spread_type}\n"
            f"Выпавшие карты:\n{cards_description}"
            f"{question_text}\n\n"
            "Дай общую интерпретацию расклада, показав:\n"
            "1. Как карты взаимодействуют друг с другом\n"
            "2. Основное послание расклада\n"
            "3. Конкретные рекомендации"
        )

        try:
            interpretation = await generate_text(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                model=self.model,
                max_tokens=self.max_tokens * 2  # Больше токенов для полного расклада
            )
            return interpretation.strip()
        except Exception as e:
            logger.error(f"Ошибка при интерпретации расклада: {e}")
            return "Извините, не удалось получить интерпретацию расклада."

    def _get_fallback_interpretation(self, drawn_card: DrawnCard) -> str:
        """Запасная интерпретация при ошибке API"""
        if drawn_card.card.card_type == CardType.MAJOR_ARCANA:
            return (
                f"Карта {drawn_card.card.name} - один из старших арканов, "
                f"символизирующий важные жизненные уроки и трансформации. "
                f"{'В перевёрнутом положении указывает на внутренние препятствия.' if drawn_card.is_reversed else 'В прямом положении - благоприятный знак.'}"
            )
        else:
            return (
                f"Карта {drawn_card.card.name} из масти {drawn_card.card.suit.value} "
                f"говорит о текущих событиях и эмоциях. "
                f"{'Перевёрнутое положение советует обратить внимание на внутренние блоки.' if drawn_card.is_reversed else 'Прямое положение обещает гармоничное развитие.'}"
            )


class TarotReading:
    """Класс для проведения гадания"""

    def __init__(self, deck: Optional[TarotDeck] = None,
                 interpreter: Optional[TarotInterpreter] = None):
        self.deck = deck or TarotDeck()
        self.interpreter = interpreter or TarotInterpreter()

    async def simple_reading(self, count: int = DEFAULT_CARD_COUNT,
                             reversed_chance: float = DEFAULT_REVERSED_CHANCE,
                             question: Optional[str] = None) -> Dict[str, any]:
        """
        Простое гадание

        Returns:
            Словарь с картами и интерпретациями
        """
        drawn_cards = self.deck.draw_cards(count, reversed_chance)

        # Получаем интерпретации для каждой карты
        interpretations = []
        for card in drawn_cards:
            interpretation = await self.interpreter.interpret_card(card, question)
            interpretations.append(interpretation)

        # Получаем общую интерпретацию
        overall_interpretation = await self.interpreter.interpret_spread(
            drawn_cards, "simple", question
        )

        return {
            "cards": drawn_cards,
            "individual_interpretations": interpretations,
            "overall_interpretation": overall_interpretation,
            "question": question
        }

    async def three_card_spread(self,
                                reversed_chance: float = DEFAULT_REVERSED_CHANCE,
                                question: Optional[str] = None) -> Dict[str, any]:
        """
        Расклад на три карты: Прошлое - Настоящее - Будущее
        """
        positions = ["Прошлое", "Настоящее", "Будущее"]
        drawn_cards = self.deck.draw_cards(3, reversed_chance, positions)

        interpretations = []
        for card in drawn_cards:
            interpretation = await self.interpreter.interpret_card(card, question)
            interpretations.append(interpretation)

        overall_interpretation = await self.interpreter.interpret_spread(
            drawn_cards, "Прошлое-Настоящее-Будущее", question
        )

        return {
            "cards": drawn_cards,
            "individual_interpretations": interpretations,
            "overall_interpretation": overall_interpretation,
            "question": question,
            "spread_type": "Прошлое-Настоящее-Будущее"
        }

    async def relationship_spread(self,
                                  reversed_chance: float = DEFAULT_REVERSED_CHANCE,
                                  question: Optional[str] = None) -> Dict[str, any]:
        """
        Расклад на отношения
        """
        positions = [
            "Вы",
            "Партнёр",
            "Основа отношений",
            "Препятствия",
            "Потенциал"
        ]
        drawn_cards = self.deck.draw_cards(5, reversed_chance, positions)

        interpretations = []
        for card in drawn_cards:
            interpretation = await self.interpreter.interpret_card(card, question)
            interpretations.append(interpretation)

        overall_interpretation = await self.interpreter.interpret_spread(
            drawn_cards, "Расклад на отношения", question
        )

        return {
            "cards": drawn_cards,
            "individual_interpretations": interpretations,
            "overall_interpretation": overall_interpretation,
            "question": question,
            "spread_type": "Расклад на отношения"
        }


# Функции для обратной совместимости
def draw_cards(cards_dict: Dict[str, str], count: int = 3,
               reversed_chance: float = DEFAULT_REVERSED_CHANCE) -> List[Tuple[str, str, bool]]:
    """
    Обратная совместимость со старым API
    """
    deck = TarotDeck()
    drawn_cards = deck.draw_cards(count, reversed_chance)

    return [
        (card.card.filename, card.card.name, card.is_reversed)
        for card in drawn_cards
    ]


async def get_card_interpretation(card_name: str, position: str,
                                  is_reversed: bool) -> str:
    """
    Обратная совместимость со старым API
    """
    deck = TarotDeck()
    interpreter = TarotInterpreter()

    # Находим карту по имени
    card = deck.get_card_by_name(card_name)
    if not card:
        return f"Карта '{card_name}' не найдена в колоде."

    drawn_card = DrawnCard(
        card=card,
        is_reversed=is_reversed,
        position=position
    )

    return await interpreter.interpret_card(drawn_card)


# Вспомогательные функции
def get_card_image_path(card: TarotCard) -> Path:
    """Получить путь к изображению карты"""
    return TARO_FOLDER / card.filename


def validate_image_files() -> List[str]:
    """
    Проверить наличие файлов изображений

    Returns:
        Список отсутствующих файлов
    """
    deck = TarotDeck()
    missing_files = []

    for card in deck.cards.values():
        image_path = get_card_image_path(card)
        if not image_path.exists():
            missing_files.append(str(image_path))

    return missing_files


# Создаём глобальную переменную для обратной совместимости
deck = TarotDeck()
manara_cards = {card.filename: card.name for card in deck.cards.values()}

# Экспортируем переменную REVERSED_CHANCE для обратной совместимости
REVERSED_CHANCE = DEFAULT_REVERSED_CHANCE


# Пример использования
async def main():
    """Пример использования модуля"""
    reading = TarotReading()

    # Простое гадание на 3 карты
    result = await reading.simple_reading(
        count=3,
        question="Что меня ждёт в ближайшем будущем?"
    )

    print("Выпавшие карты:")
    for i, card in enumerate(result["cards"]):
        print(f"{i + 1}. {card.full_name}")

    print("\nИнтерпретации:")
    for i, interp in enumerate(result["individual_interpretations"]):
        print(f"\nКарта {i + 1}:")
        print(interp)

    print("\nОбщая интерпретация:")
    print(result["overall_interpretation"])


if __name__ == "__main__":
    asyncio.run(main())
