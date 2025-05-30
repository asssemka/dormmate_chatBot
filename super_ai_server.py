from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import logging
from typing import Optional
import time
import re
from perfect_database import PerfectDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🎯 ИДЕАЛЬНО ТОЧНЫЙ AI ЧатБот", version="7.0.0")


class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    max_tokens: int = 100


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    confidence: float
    model_info: str
    generation_time: float


class PerfectChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.database = PerfectDatabase()
        self.load_model()

    def load_model(self):
        """Загружает модель для неизвестных вопросов"""
        try:
            model_path = "./fixed_super_model"
            base_model_name = "ai-forever/rugpt3small_based_on_gpt2"

            print(f"🧠 Загружаем модель с устройства: {self.device}")

            import os
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                print("🎯 Загружаем ТОЧНУЮ МОДЕЛЬ...")

                # Загружаем токенизатор из базовой модели
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                # Загружаем базовую модель
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )

                # Загружаем LoRA адаптер
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.model = self.model.to(self.device)

                print("✅ ТОЧНАЯ МОДЕЛЬ загружена!")

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
            logger.info("⚠️ Продолжаем работу только с базой данных")

    def clean_response(self, text: str) -> str:
        """Очистка ответа от артефактов"""
        # Убираем артефакты
        artifacts = [
            r'<\|.*?\|>', r'<.*?>', r'\|.*?\|',
            r'<endoftext>', r'<startoftext>', r'<pad>',
            r'endof.*?', r'enabled.*?', r'Beta.*?',
            r'Factory.*?', r'Container.*?', r'\[.*?\]'
        ]

        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)

        # Убираем лишние символы и пробелы
        text = re.sub(r'[^\w\s.,!?:;№()-]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def generate_model_response(self, question: str, max_tokens: int = 100) -> tuple:
        """Генерирует ответ с помощью модели"""
        if self.model is None or self.tokenizer is None:
            return "Модель не загружена. Используется только база данных.", 0.0, 0.0

        start_time = time.time()

        try:
            # Промпт
            prompt = f"Вопрос: {question}\nОтвет:"

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
            if "Ответ:" in response:
                answer = response.split("Ответ:")[-1].strip()
            else:
                answer = response.replace(prompt, "").strip()

            # Очищаем ответ
            answer = self.clean_response(answer)

            generation_time = time.time() - start_time

            return answer, 0.5, generation_time

        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return "Извините, произошла техническая ошибка. Попробуйте позже.", 0.1, 0.0

    def get_perfect_answer(self, question: str, max_tokens: int = 100) -> tuple:
        """Получает идеально точный ответ"""
        start_time = time.time()

        # Сначала ищем в базе данных
        db_answer, confidence = self.database.get_answer(question)

        # Если нашли с высокой уверенностью, возвращаем
        if confidence >= 0.7:
            generation_time = time.time() - start_time
            return db_answer, confidence, generation_time

        # Если не нашли в базе, используем модель
        model_answer, model_confidence, model_time = self.generate_model_response(question, max_tokens)

        # Если ответ из базы лучше, используем его
        if confidence > model_confidence:
            generation_time = time.time() - start_time
            return db_answer, confidence, generation_time

        # Иначе используем ответ модели
        return model_answer, model_confidence, model_time


# Создаем бота
chatbot = PerfectChatbot()


@app.get("/")
def root():
    import os
    model_type = "ИДЕАЛЬНО ТОЧНАЯ МОДЕЛЬ" if os.path.exists("./fixed_super_model") else "База данных + Базовая модель"
    return {
        "message": "🎯 ИДЕАЛЬНО ТОЧНЫЙ AI ЧатБот работает!",
        "model_info": model_type,
        "device": str(chatbot.device),
        "university": "Алматы",
        "database_size": len(chatbot.database.qa_pairs),
        "status": "ready"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Отвечает на вопросы с идеальной точностью"""
    try:
        logger.info(f"📝 Вопрос от {request.user_id}: {request.question}")

        answer, confidence, gen_time = chatbot.get_perfect_answer(
            request.question,
            request.max_tokens
        )

        import os
        model_info = "perfect_hybrid_model" if os.path.exists("./fixed_super_model") else "perfect_database"

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
