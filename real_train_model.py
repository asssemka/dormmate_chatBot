import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os

class RealStudentChatbotTrainer:
    def __init__(self):
        # Используем легкую русскую модель
        self.model_name = "ai-forever/rugpt3small_based_on_gpt2"
        self.output_dir = "./real_student_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_dataset(self):
        """Загружает реальный датасет"""
        print("📚 Загружаем РЕАЛЬНЫЙ датасет...")
        
        data = []
        with open("real_dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    # Форматируем для обучения
                    text = f"Студент: {item['question']}\nПомощник: {item['answer']}<|endoftext|>"
                    data.append({"text": text})
        
        print(f"✅ Загружено {len(data)} реальных примеров")
        return Dataset.from_list(data)
    
    def setup_model_and_tokenizer(self):
        """Настраивает модель и токенизатор"""
        print("🧠 Настраиваем русскую модель...")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Добавляем специальные токены
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # Настраиваем LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ Модель настроена для реальных данных!")
    
    def tokenize_function(self, examples):
        """Токенизирует примеры"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=300,  # Увеличили для длинных ответов
            return_tensors="pt"
        )
    
    def train(self):
        """Обучает модель на реальных данных"""
        print("🚀 Начинаем обучение на РЕАЛЬНЫХ данных...")
        
        # Загружаем данные
        dataset = self.load_dataset()
        
        # Настраиваем модель
        self.setup_model_and_tokenizer()
        
        # Токенизируем данные
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=8,  # Больше эпох для лучшего качества
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            logging_steps=10,
            save_steps=200,
            evaluation_strategy="no",
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            learning_rate=5e-4,
            fp16=False,
        )
        
        # Коллатор данных
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Создаем тренера
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Обучаем!
        print("🎯 Запускаем обучение на реальных данных...")
        trainer.train()
        
        # Сохраняем модель
        print("💾 Сохраняем обученную модель...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("🎉 Обучение на реальных данных завершено!")
        print(f"📁 Модель сохранена в: {self.output_dir}")

def main():
    print("🎓 ОБУЧЕНИЕ НА РЕАЛЬНЫХ ДАННЫХ УНИВЕРСИТЕТА")
    print("=" * 60)
    
    trainer = RealStudentChatbotTrainer()
    trainer.train()
    
    print("\n✅ Готово! Теперь запустите сервер с реальной моделью")

if __name__ == "__main__":
    main()
