import requests
import time


def test_smart_chatbot():
    """Тестирует умный чатбот с разговорным языком"""
    base_url = "http://localhost:8000"

    print("🧠 ТЕСТИРУЕМ УМНЫЙ AI ЧАТБОТ С РАЗГОВОРНЫМ ЯЗЫКОМ")
    print("=" * 80)

    # Проверяем статус
    try:
        response = requests.get(f"{base_url}/")
        info = response.json()
        print("📊 ИНФОРМАЦИЯ ОБ УМНОЙ МОДЕЛИ:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return

    # Тестовые вопросы с разговорным языком
    test_questions = [
        # Стандартные вопросы
        "Где находится общежитие ДС3?",
        "Где находится общежитие ДС2а?",
        "Мне не дали общежитие",

        # Разговорные варианты с "общага"
        "Где ДС3?",
        "Че за адрес у ДС2а?",
        "Куда ехать если дали место в ДС3?",
        "Не дали общагу, че делать?",
        "Как подать заявку на общагу?",
        "Че нужно для общаги?",
        "Какие общаги есть?",
        "Список всех общаг",
        "Сколько общаг в универе?",

        # Варианты с "дом студентов"
        "Где дом студентов 3?",
        "Адрес студдома 2а?",
        "Как попасть в студенческий дом?",

        # Сокращения и склонения
        "Где дс 3?",
        "Адрес дс-2б?",
        "В какой общаге лучше?",
        "Общагах каких адреса?",
        "Общежитий сколько?",
        "Жилье студенческое где?",

        # Новые вопросы
        "Как подать апелляцию на отказ в общаге?",
        "Сколько мест в комнатах общаг?",
        "Че можно взять с собой в общагу?",
        "Правила проживания в общагах",
        "Стоимость общаги",

        # Вопросы для стоп-фраз
        "Привет как дела?",
        "Какая погода?",
        "Где библиотека?",
        "Расскажи анекдот",
        "Сколько времени?",
        "Что на обед?",

        # Очень короткие вопросы
        "Че?",
        "Что?",
        "Как?",
        "Общага?",
        "",
        "   "
    ]

    print(f"\n🧪 ТЕСТИРУЕМ {len(test_questions)} УМНЫХ ВОПРОСОВ:")
    print("=" * 80)

    results = {
        "database_answers": 0,
        "model_answers": 0,
        "stop_phrases": 0,
        "high_confidence": 0,
        "total_time": 0
    }

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 УМНЫЙ ТЕСТ {i}/{len(test_questions)}")
        print(f"❓ Вопрос: '{question}'")

        try:
            start = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "question": question,
                    "user_id": f"smart_user_{i}",
                    "max_tokens": 100
                },
                timeout=30
            )
            end = time.time()

            request_time = end - start
            results["total_time"] += request_time

            if response.status_code == 200:
                result = response.json()

                print(f"🤖 Ответ: {result['answer']}")
                print(f"📊 Уверенность: {result['confidence']:.2f}")
                print(f"⏱️ Время генерации: {result['generation_time']:.2f}с")
                print(f"🔧 Модель: {result['model_info']}")
                print(f"📡 Источник: {result['source']}")
                print(f"⏰ Общее время: {request_time:.2f}с")

                # Статистика
                if result['source'] == 'database':
                    results["database_answers"] += 1
                else:
                    results["model_answers"] += 1

                if result['confidence'] > 0.8:
                    results["high_confidence"] += 1

                # Проверяем на стоп-фразы
                stop_keywords = ["не знаю", "нет информации", "выходит за рамки", "не могу ответить"]
                if any(keyword in result['answer'].lower() for keyword in stop_keywords):
                    results["stop_phrases"] += 1
                    print("🛑 СТОП-ФРАЗА обнаружена")

                # Оценка качества
                if result['confidence'] > 0.8:
                    print("🌟 ОТЛИЧНОЕ качество")
                elif result['confidence'] > 0.6:
                    print("✅ ХОРОШЕЕ качество")
                elif result['confidence'] > 0.3:
                    print("⚠️ СРЕДНЕЕ качество")
                else:
                    print("❌ НИЗКОЕ качество")

            else:
                print(f"❌ HTTP Ошибка: {response.status_code}")

        except Exception as e:
            print(f"❌ Ошибка: {e}")

        print("-" * 80)

    # Итоговая статистика
    print(f"\n🧠 СТАТИСТИКА УМНОГО ЧАТБОТА:")
    print("=" * 80)
    print(f"📊 Всего вопросов: {len(test_questions)}")
    print(f"🗃️ Ответов из базы данных: {results['database_answers']}")
    print(f"🤖 Ответов от модели: {results['model_answers']}")
    print(f"🛑 Стоп-фраз: {results['stop_phrases']}")
    print(f"🌟 Высокая уверенность: {results['high_confidence']}")
    print(f"⏱️ Среднее время: {results['total_time'] / len(test_questions):.2f}с")

    # Оценка работы
    database_percentage = (results['database_answers'] / len(test_questions)) * 100
    stop_percentage = (results['stop_phrases'] / len(test_questions)) * 100

    print(f"\n📈 АНАЛИЗ:")
    print(f"🗃️ База данных: {database_percentage:.1f}%")
    print(f"🛑 Стоп-фразы: {stop_percentage:.1f}%")

    if stop_percentage > 20:
        print("✅ Хорошо использует стоп-фразы для неизвестных вопросов")

    if database_percentage > 60:
        print("✅ Хорошо работает с базой данных")

    print("\n🎉 ТЕСТИРОВАНИЕ УМНОГО ЧАТБОТА ЗАВЕРШЕНО!")


if __name__ == "__main__":
    test_smart_chatbot()
