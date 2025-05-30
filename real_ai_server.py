from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import logging
from typing import Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🏫 Университетский AI ЧатБот", version="4.0.0")

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    max_tokens: int = 150

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    model_info: str
    generation_time: float

class UniversityAIChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Загружает обученную модель"""
        try:
            model_path = "./real_student_model"
            base_model_name = "ai-forever/rugpt3small_based_on_gpt2"
            
            print(f"🧠 Загружаем модель с устройства: {self.device}")
            
            import os
            if os.path.exists(model_path):
                print("📚 Загружаем ОБУЧЕННУЮ модель на реальных данных...")
                
                # Загружаем токенизатор
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Загружаем базовую модель
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                
                # Загружаем LoRA адаптер
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.model = self.model.to(self.device)
                
                print("✅ Обученная модель загружена!")
                
            else:
                print("⚠️ Обученная модель не найдена, используем базовую...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model = self.model.to(self.device)
                
                print("✅ Базовая модель загружена!")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise e
    
    def clean_response(self, text: str) -> str:
        """Очищает ответ от артефактов"""
        # Убираем повторения и лишние символы
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Убираем возможные повторения фраз
        sentences = text.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def calculate_confidence(self, question: str, answer: str) -> float:
        """Вычисляет уверенность в ответе"""
        # Простая эвристика для оценки качества ответа
        confidence = 0.5
        
        # Проверяем длину ответа
        if len(answer) > 20:
            confidence += 0.2
        
        # Проверяем наличие ключевых слов
        university_keywords = ['общежитие', 'деканат', 'университет', 'студент', 'заявка', 'адрес', 'алматы']
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        for keyword in university_keywords:
            if keyword in question_lower and keyword in answer_lower:
                confidence += 0.1
        
        # Проверяем на наличие адреса
        if 'алматы' in answer_lower and ('мкр' in answer_lower or 'таугуль' in answer_lower):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def generate_response(self, question: str, max_tokens: int = 150) -> tuple:
        """Генерирует ответ на вопрос"""
        import time
        
        start_time = time.time()
        
        try:
            # Форматируем промпт как в обучении
            prompt = f"Студент: {question}\nПомощник:"
            
            # Токенизируем
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=200
            ).to(self.device)
            
            # Генерируем с оптимизированными параметрами
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Декодируем
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем ответ
            if "Помощник:" in response:
                answer = response.split("Помощник:")[-1].strip()
            else:
                answer = response.replace(prompt, "").strip()
            
            # Очищаем ответ
            answer = self.clean_response(answer)
            
            # Если ответ пустой или слишком короткий
            if not answer or len(answer) < 5:
                answer = "Извините, я не смог найти точный ответ на ваш вопрос. Рекомендую обратиться в деканат или отдел студенческого проживания."
            
            # Вычисляем уверенность
            confidence = self.calculate_confidence(question, answer)
            
            generation_time = time.time() - start_time
            
            return answer, confidence, generation_time
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return "Извините, произошла техническая ошибка. Попробуйте позже.", 0.1, 0

# Создаем экземпляр бота
chatbot = UniversityAIChatbot()

@app.get("/")
def root():
    import os
    model_type = "Обученная на реальных данных" if os.path.exists("./real_student_model") else "Базовая модель"
    return {
        "message": "🏫 Университетский AI ЧатБот работает!",
        "model_info": model_type,
        "device": str(chatbot.device),
        "university": "Алматы",
        "status": "ready"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Отвечает на вопросы студентов"""
    try:
        logger.info(f"📝 Вопрос от {request.user_id}: {request.question}")
        
        answer, confidence, gen_time = chatbot.generate_response(
            request.question, 
            request.max_tokens
        )
        
        import os
        model_info = "real_trained_model" if os.path.exists("./real_student_model") else "base_model"
        
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            model_info=model_info,
            generation_time=gen_time
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
