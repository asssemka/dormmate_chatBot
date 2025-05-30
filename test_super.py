import requests
import time


def test_perfect_chatbot():
    """Тестирует идеально точный чатбот"""
    base_url = "http://localhost:8000"

    print("🎯 ТЕСТИРУЕМ ИДЕАЛЬНО ТОЧНЫЙ AI ЧАТБОТ")
    print("=" * 70)

    # Проверяем статус
    try:
        response = requests.get(f"{base_url}/")
        info = response.json()
        print("📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return

    # Тестовые вопросы с эталонными ответами
    test_questions = [
        {
            "question": "Где находится общежитие ДС3?",
            "expected": "Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А."
        },
        {
            "question": "Где находится общежитие ДС2а?",
            "expected": "Общежитие ДС2А находится по адресу: г. Алматы, Таугуль 32."
        },
        {
            "question": "Где находится общежитие ДС2б?",
            "expected": "Общежитие ДС2б находится по адресу: г. Алматы, Таугуль 34."
        },
        {
            "question": "Где находится общежитие Емен?",
            "expected": "Общежитие Емен находится по адресу: г. Алматы, мкр №10 26/1."
        },
        {
            "question": "Мне не дали общежитие",
            "expected": "Рекомендую обратиться в деканат или отдел студенческого проживания для консультации и возможного пересмотра заявки."
        },
        {
            "question": "Как подать заявку на общежитие?",
            "expected": "Подать заявку можно через личный кабинет на сайте, заполнив анкету и приложив необходимые документы."
        },
        {
            "question": "Какие документы нужны для общежития?",
            "expected": "Потребуется удостоверение личности, справка о доходах семьи и справка о зачислении в университет."
        },
        {
            "question": "Как оплатить общежитие?",
            "expected": "Оплату можно произвести через личный кабинет или банковский перевод по реквизитам университета."
        },
        {
            "question": "Что делать, если отказали в общежитии?",
            "expected": "Можно подать апелляцию или уточнить наличие мест в других общежитиях через администрацию."
        },
        {
            "question": "Можно ли поселиться вдвоём с другом?",
            "expected": "При подаче заявки укажите пожелания по соседству. Администрация учтёт ваши предпочтения при расселении."
        },
        {
            "question": "Сколько человек живёт в комнате?",
            "expected": "Как правило, в одной комнате проживают от двух до четырёх человек в зависимости от общежития и категории комнаты."
        },
        {
            "question": "Адрес ДС3 в Алматы",
            "expected": "Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А."
        }
    ]

    print(f"\n🧪 ТЕСТИРУЕМ {len(test_questions)} ВОПРОСОВ:")
    print("=" * 70)

    perfect_answers = 0
    good_answers = 0
    total_confidence = 0

    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        expected = test_case["expected"]

        print(f"\n📝 ТЕСТ {i}/{len(test_questions)}")
        print(f"❓ Вопрос: {question}")
        print(f"🎯 Ожидаемый: {expected}")

        try:
            start = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "question": question,
                    "user_id": f"perfect_student_{i}",
                    "max_tokens": 80
                },
                timeout=30
            )
            end = time.time()

            if response.status_code == 200:
                result = response.json()

                print(f"🤖 Ответ: {result['answer']}")
                print(f"📊 Уверенность: {result['confidence']:.2f}")
                print(f"⏱️ Время генерации: {result['generation_time']:.2f}с")
                print(f"🔧 Модель: {result['model_info']}")
                print(f"📡 Общее время: {end - start:.2f}с")

                total_confidence += result['confidence']

                # Оценка точности
                if result['answer'] == expected:
                    print("🌟 ИДЕАЛЬНО ТОЧНЫЙ ОТВЕТ!")
                    perfect_answers += 1
                elif result['answer'].lower().replace(" ", "") in expected.lower().replace(" ",
                                                                                           "") or expected.lower().replace(
                        " ", "") in result['answer'].lower().replace(" ", ""):
                    print("✅ ХОРОШИЙ ОТВЕТ")
                    good_answers += 1
                else:
                    print("⚠️ НЕТОЧНЫЙ ОТВЕТ")

            else:
                print(f"❌ HTTP Ошибка: {response.status_code}")

        except Exception as e:
            print(f"❌ Ошибка: {e}")

        print("-" * 70)

    avg_confidence = total_confidence / len(test_questions) if test_questions else 0

    print(f"\n🎯 РЕЗУЛЬТАТЫ ИДЕАЛЬНО ТОЧНОЙ МОДЕЛИ:")
    print(f"🌟 Идеально точных ответов: {perfect_answers}/{len(test_questions)}")
    print(f"✅ Хороших ответов: {good_answers}/{len(test_questions)}")
    print(f"📊 Средняя уверенность: {avg_confidence:.2f}")
    print(f"🔥 Процент идеальных: {(perfect_answers / len(test_questions) * 100):.1f}%")
    print(f"📈 Общий успех: {((perfect_answers + good_answers) / len(test_questions) * 100):.1f}%")

    if perfect_answers == len(test_questions):
        print("\n🎉 МОДЕЛЬ РАБОТАЕТ ИДЕАЛЬНО! 100% ТОЧНОСТЬ!")
    elif perfect_answers >= len(test_questions) * 0.9:
        print("\n🎯 МОДЕЛЬ РАБОТАЕТ ПОЧТИ ИДЕАЛЬНО!")
    else:
        print("\n⚠️ МОДЕЛЬ НУЖДАЕТСЯ В ДОРАБОТКЕ")


if __name__ == "__main__":
    test_perfect_chatbot()
