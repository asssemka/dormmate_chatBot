import requests
import time


def interactive_test():
    """Интерактивный тест чатбота"""
    base_url = "http://localhost:8000"

    print("💬 ИНТЕРАКТИВНЫЙ ТЕСТ ЧАТБОТА")
    print("=" * 50)
    print("Введите 'quit' или 'exit' для выхода")
    print("Введите 'help' для списка примеров вопросов")
    print("=" * 50)

    # Проверяем сервер
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print(f"⚠️ Проблемы с сервером: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        print("💡 Запустите сервер: python perfect_server.py")
        return

    print("✅ Сервер доступен. Можете задавать вопросы!")

    question_count = 0

    while True:
        try:
            print(f"\n💬 Вопрос #{question_count + 1}:")
            question = input("❓ Ваш вопрос: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'выход']:
                print("👋 До свидания!")
                break

            if question.lower() == 'help':
                print("\n📝 ПРИМЕРЫ ВОПРОСОВ:")
                examples = [
                    "Где находится общежитие ДС3?",
                    "Как подать заявку на общежитие?",
                    "Какие документы нужны?",
                    "Мне не дали общежитие",
                    "Сколько человек в комнате?",
                    "Как подать апелляцию на отказ?",
                    "Куда ехать, если дали место в ДС3?",
                    "Какие общаги есть?",
                    "Че можно взять с собой?"
                ]
                for i, example in enumerate(examples, 1):
                    print(f"   {i}. {example}")
                continue

            # Отправляем вопрос
            start = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "question": question,
                    "user_id": "interactive_user",
                    "max_tokens": 120
                },
                timeout=30
            )
            end = time.time()

            if response.status_code == 200:
                result = response.json()

                print(f"\n🤖 Ответ: {result['answer']}")
                print(f"📊 Уверенность: {result['confidence']:.2f}")
                print(f"⏱️ Время: {result['generation_time']:.2f}с (запрос: {end - start:.2f}с)")
                print(f"🔧 Модель: {result['model_info']}")

                # Простая оценка качества
                if result['confidence'] > 0.8:
                    print("🌟 Высокое качество ответа")
                elif result['confidence'] > 0.6:
                    print("✅ Хорошее качество ответа")
                elif result['confidence'] > 0.4:
                    print("⚠️ Среднее качество ответа")
                else:
                    print("❌ Низкое качество ответа")

            else:
                print(f"❌ Ошибка сервера: {response.status_code}")
                print(f"📄 Ответ: {response.text}")

            question_count += 1

        except KeyboardInterrupt:
            print("\n👋 Прерывание пользователем. До свидания!")
            break

        except Exception as e:
            print(f"❌ Ошибка: {e}")

    print(f"\n📊 Всего задано вопросов: {question_count}")
    print("🏁 Интерактивный тест завершен!")


if __name__ == "__main__":
    interactive_test()
