from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import logging
from typing import Optional
import time
import re
from smart_database import SmartDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🧠 УМНЫЙ AI ЧатБот с Разговорным Языком", version="8.0.0")


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
    source: str  # "database" или "model"


class SmartChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.database = SmartDatabase()
        self.load_model()

    def load_model(self):
        try:
            model_path = "./smart_model"
            base_model_name = "ai-forever/rugpt3small_based_on_gpt2"
            print(f"Загружаем умную модель с устройства: {self.device}")
            import os
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                print("Загружаем УМНУЮ ОБУЧЕННУЮ МОДЕЛЬ...")
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.model = self.model.to(self.device)
                print("УМНАЯ МОДЕЛЬ загружена!")
            else:
                print("Обученная модель не найдена, используем базовую...")
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model = self.model.to(self.device)
                print("Базовая модель загружена!")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.info(" Продолжаем работу только с умной базой данных")

    def clean_response(self, text: str) -> str:
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

        # Берем только первое предложение если оно содержательное
        sentences = text.split('.')
        if sentences and len(sentences[0].strip()) > 10:
            result = sentences[0].strip() + '.'
        else:
            result = text.strip()

        return result

    def generate_model_response(self, question: str, max_tokens: int = 100) -> tuple:
        """Генерирует ответ с помощью модели"""
        if self.model is None or self.tokenizer is None:
            return "Модель не загружена.", 0.0, 0.0

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

            return answer, 0.6, generation_time

        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return "Извините, произошла техническая ошибка.", 0.1, 0.0

    def get_smart_answer(self, question: str, max_tokens: int = 100) -> tuple:
        """Получает умный ответ с использованием базы данных и модели"""
        start_time = time.time()

        # Сначала пробуем умную базу данных
        db_answer, db_confidence = self.database.get_smart_answer(question)

        # Если база данных дала хороший ответ, используем его
        if db_confidence >= 0.7:
            generation_time = time.time() - start_time
            return db_answer, db_confidence, generation_time, "database"

        # Если база данных дала стоп-фразу, проверяем модель
        if db_confidence <= 0.2 and self.model is not None:
            model_answer, model_confidence, model_time = self.generate_model_response(question, max_tokens)

            # Если модель дала лучший ответ
            if model_confidence > db_confidence and len(model_answer) > 10:
                return model_answer, model_confidence, model_time, "model"

        # Иначе используем ответ из базы данных (включая стоп-фразы)
        generation_time = time.time() - start_time
        return db_answer, db_confidence, generation_time, "database"


# Создаем умного бота
chatbot = SmartChatbot()


@app.get("/")
def root():
    import os
    model_type = "УМНАЯ МОДЕЛЬ" if os.path.exists("./smart_model") else "Умная база данных + Базовая модель"
    return {
        "message": "🧠 УМНЫЙ AI ЧатБот с разговорным языком работает!",
        "model_info": model_type,
        "device": str(chatbot.device),
        "university": "Алматы",
        "database_size": len(chatbot.database.qa_pairs),
        "features": ["разговорный язык", "стоп-фразы", "умный поиск"],
        "status": "ready"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Отвечает на вопросы с пониманием разговорного языка"""
    try:
        logger.info(f"📝 Умный вопрос от {request.user_id}: {request.question}")

        answer, confidence, gen_time, source = chatbot.get_smart_answer(
            request.question,
            request.max_tokens
        )

        import os
        model_info = "smart_hybrid_model" if os.path.exists("./smart_model") else "smart_database"

        return ChatResponse(
            answer=answer,
            confidence=confidence,
            model_info=model_info,
            generation_time=gen_time,
            source=source
        )

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
