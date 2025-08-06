import openai

#API_KEY = ""


def list_available_models(api_key: str):
    openai.api_key = api_key
    try:
        response = openai.Model.list()
        # Выводим идентификаторы (id) всех моделей
        for model in response["data"]:
            print(model["id"])
    except Exception as e:
        print("Ошибка при запросе списка моделей:", e)


if __name__ == "__main__":
    list_available_models(API_KEY)
g
