import requests
import time

def test_real_chatbot():
    """Тестирует чатбот на реальных данных"""
    base_url = "http://localhost:8000"
    
    print("🏫 ТЕСТИРУЕМ УНИВЕРСИТЕТСКИЙ AI ЧАТБОТ")
    print("=" * 60)
    
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
    
    # Реальные тестовые вопросы
    real_questions = [
        "Где находится общежитие ДС3?",
        "Где находится общежитие ДС2а?", 
        "Где находится общежитие Емен?",
        "Мне не дали общежитие",
        "Как подать заявку на общежитие?",
        "Какие документы нужны для общежития?",
        "Как оплатить общежитие?",
        "Что делать, если отказали в общежитии?",
        "Можно ли поселиться вдвоём с другом?",
        "Сколько человек живёт в комнате?",
        "Привет, как дела?",  # Тест на общие вопросы
        "Где библиотека?"     # Тест на неизвестные вопросы
    ]
    
    print(f"\n🧪 ТЕСТИРУЕМ {len(real_questions)} РЕАЛЬНЫХ ВОПРОСОВ:")
    print("=" * 60)
    
    successful_answers = 0
    high_confidence_answers = 0
    
    for i, question in enumerate(real_questions, 1):
        print(f"\n📝 ТЕСТ {i}/{len(real_questions)}")
        print(f"❓ Вопрос: {question}")
        
        try:
            start = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "question": question,
                    "user_id": f"student_{i}",
                    "max_tokens": 120
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
                print(f"📡 Общее время: {end-start:.2f}с")
                
                # Оценка качества
                if len(result['answer']) > 30 and result['confidence'] > 0.6:
                    print("✅ ОТЛИЧНЫЙ ОТВЕТ")
                    successful_answers += 1
                    if result['confidence'] > 0.8:
                        high_confidence_answers += 1
                elif len(result['answer']) > 15:
                    print("⚠️ СРЕДНИЙ ОТВЕТ")
                    successful_answers += 0.5
                else:
                    print("❌ СЛАБЫЙ ОТВЕТ")
                    
            else:
                print(f"❌ HTTP Ошибка: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print("-" * 60)
    
    print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"✅ Успешных ответов: {successful_answers}/{len(real_questions)}")
    print(f"🎯 Высокая уверенность: {high_confidence_answers}/{len(real_questions)}")
    print(f"📈 Процент успеха: {(successful_answers/len(real_questions)*100):.1f}%")
    print(f"🔥 Процент отличных ответов: {(high_confidence_answers/len(real_questions)*100):.1f}%")

if __name__ == "__main__":
    test_real_chatbot()
