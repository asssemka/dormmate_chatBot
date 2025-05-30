from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🤖 AI Студенческий ЧатБот", version="2.0.0")

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    max_tokens: int = 100

class ChatResponse(BaseModel):
    answer: str
    model_type: str
    generation_time: float

class AIStudentChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Загружает обученную модель"""
        try:
            model_path = "./trained_student_model"
            base_model_name = "microsoft/DialoGPT-small"
            
            print(f"🧠 Загружаем модель с устройства: {self.device}")
            
            # Проверяем есть ли обученная модель
            import os
            if os.path.exists(model_path):
                print("📚 Загружаем ОБУЧЕННУЮ модель...")
                
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
    
    def generate_response(self, question: str, max_tokens: int = 100) -> tuple:
        """Генерирует ответ на вопрос"""
        import time
        
        start_time = time.time()
        
        try:
            # Форматируем промпт
            prompt = f"Пользователь: {question}\nАссистент:"
            
            # Токенизируем
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=200
            ).to(self.device)
            
            # Генерируем
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Декодируем
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем ответ
            if "Ассистент:" in response:
                answer = response.split("Ассистент:")[-1].strip()
            else:
                answer = response.replace(prompt, "").strip()
            
            generation_time = time.time() - start_time
            
            return answer, generation_time
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return f"Извините, произошла ошибка: {str(e)}", 0

# Создаем экземпляр бота
chatbot = AIStudentChatbot()

@app.get("/")
def root():
    model_type = "Обученная модель" if "./trained_student_model" else "Базовая модель"
    return {
        "message": "🤖 AI Студенческий ЧатБот работает!",
        "model_type": model_type,
        "device": str(chatbot.device),
        "status": "ready"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Генерирует ответ с помощью AI"""
    try:
        logger.info(f"📝 Вопрос от {request.user_id}: {request.question}")
        
        answer, gen_time = chatbot.generate_response(
            request.question, 
            request.max_tokens
        )
        
        model_type = "trained_model" if "./trained_student_model" else "base_model"
        
        return ChatResponse(
            answer=answer,
            model_type=model_type,
            generation_time=gen_time
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
