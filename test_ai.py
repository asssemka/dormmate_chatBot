import requests
import time

def test_ai_chatbot():
    """Тестирует AI чатбот"""
    base_url = "http://localhost:8000"
    
    print("🤖 ТЕСТИРУЕМ AI ЧАТБОТ")
    print("=" * 50)
    
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
    
    # Тестовые вопросы
    questions = [
        "Где находится общежитие дс2а?",
        "Как подать заявку на общежитие?",
        "Сколько стоит проживание?",
        "Привет, как дела?",
        "Что делать если сосед мешает?",
        "Есть ли интернет в общежитии?"
    ]
    
    print(f"\n🧪 ТЕСТИРУЕМ {len(questions)} ВОПРОСОВ:")
    print("=" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 ТЕСТ {i}/{len(questions)}")
        print(f"❓ Вопрос: {question}")
        
        try:
            start = time.time()
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "question": question,
                    "user_id": f"test_{i}",
                    "max_tokens": 80
                },
                timeout=30
            )
            end = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"🤖 Ответ: {result['answer']}")
                print(f"⏱️ Время генерации: {result['generation_time']:.2f}с")
                print(f"🔧 Тип модели: {result['model_type']}")
                print(f"📡 Общее время: {end-start:.2f}с")
                
                if len(result['answer']) > 10:
                    print("✅ ХОРОШИЙ ОТВЕТ")
                else:
                    print("⚠️ КОРОТКИЙ ОТВЕТ")
                    
            else:
                print(f"❌ HTTP Ошибка: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_ai_chatbot()
