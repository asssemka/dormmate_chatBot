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

class StudentChatbotTrainer:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"  # Легкая модель для быстрого обучения
        self.output_dir = "./trained_student_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_dataset(self):
        """Загружает и подготавливает датасет"""
        print("📚 Загружаем датасет...")
        
        data = []
        with open("dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    # Форматируем для обучения диалоговой модели
                    text = f"Пользователь: {item['instruction']}\nАссистент: {item['output']}<|endoftext|>"
                    data.append({"text": text})
        
        print(f"✅ Загружено {len(data)} примеров")
        return Dataset.from_list(data)
    
    def setup_model_and_tokenizer(self):
        """Настраивает модель и токенизатор"""
        print("🧠 Настраиваем модель...")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Настраиваем LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"]  # Для DialoGPT
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ Модель настроена!")
    
    def tokenize_function(self, examples):
        """Токенизирует примеры"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
    
    def train(self):
        """Обучает модель"""
        print("🚀 Начинаем обучение...")
        
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
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            learning_rate=5e-5,
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
        print("🎯 Запускаем обучение...")
        trainer.train()
        
        # Сохраняем модель
        print("💾 Сохраняем модель...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("🎉 Обучение завершено!")
        print(f"📁 Модель сохранена в: {self.output_dir}")

def main():
    print("🎓 ОБУЧЕНИЕ СТУДЕНЧЕСКОГО ЧАТБОТА")
    print("=" * 50)
    
    trainer = StudentChatbotTrainer()
    trainer.train()
    
    print("\n✅ Готово! Теперь можете запустить сервер с обученной моделью")

if __name__ == "__main__":
    main()
