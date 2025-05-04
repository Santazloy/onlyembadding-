# exchange.py
"""
Модуль с логикой валютного конвертера.
Не содержит код aiogram-хендлеров: просто функции для получения курсов
и форматирования результата.
"""

import logging
import requests

logger = logging.getLogger(__name__)

CURRENCIES = {
    'RUB': {
        'name': 'Рубль',
        'flag': '🇷🇺'
    },
    'USD': {
        'name': 'Доллар',
        'flag': '🇺🇸'
    },
    'UAH': {
        'name': 'Гривна',
        'flag': '🇺🇦'
    },
    'CNY': {
        'name': 'Юань',
        'flag': '🇨🇳'
    },
    'EUR': {
        'name': 'Евро',
        'flag': '🇪🇺'
    },
    'USDT': {
        'name': 'Tether',
        'flag': '🏴‍☠️'
    },
}


def get_usdt_rate_coingecko():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=usd"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info("CoinGecko response for USDT: %s", data)
            usd_value = data.get("tether", {}).get("usd")
            if usd_value and usd_value != 0:
                return 1.0 / float(usd_value)
    except Exception as e:
        logger.error("Ошибка при получении курса USDT с CoinGecko: %s", e)
    return None


def get_fiat_rates():
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info("open.er-api.com response: %s", data)
            rates = data.get("rates", {})
            filtered = {
                key: rates.get(key)
                for key in ["RUB", "UAH", "CNY", "EUR"]
            }
            filtered["USD"] = 1.0
            logger.info("Фиатные курсы: %s", filtered)
            return filtered
    except Exception as e:
        logger.error("Ошибка при получении курсов с open.er-api.com: %s", e)
    return {}


def get_all_rates(base_currency: str):
    rates = get_fiat_rates()
    usdt_rate = get_usdt_rate_coingecko()
    if usdt_rate is not None:
        rates["USDT"] = usdt_rate
    if not rates:
        return {}

    base_currency = base_currency.upper()
    if base_currency == "USD":
        return rates

    if base_currency in rates:
        base_to_usd = 1.0 / rates[base_currency]
        converted = {}
        for code, r in rates.items():
            converted[code] = base_to_usd * r
        return converted

    return {}


def convert_and_format(amount: float, base_currency: str) -> str:
    base_currency = base_currency.upper()
    rates = get_all_rates(base_currency)
    if not rates:
        return "<b>❌ Не удалось получить курсы валют.</b>"

    result_lines = [f"💱 {amount} {base_currency} ="]
    for code, rate in rates.items():
        if code == base_currency:
            continue
        converted = round(amount * rate, 2)
        flag = CURRENCIES.get(code, {}).get("flag", "")
        name = CURRENCIES.get(code, {}).get("name", code)
        result_lines.append(f"{flag} {converted} {code} — {name}")

    formatted = "<b>" + "\n".join(result_lines) + "</b>"
    return formatted
