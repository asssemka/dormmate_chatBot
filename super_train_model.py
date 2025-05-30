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


class FixedSuperTrainer:
    def __init__(self):
        # Используем проверенную модель
        self.model_name = "ai-forever/rugpt3small_based_on_gpt2"  # Возвращаемся к рабочей
        self.output_dir = "./fixed_super_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self):
        """Загружает датасет с правильным форматированием"""
        print("📚 Загружаем исправленный датасет...")

        data = []
        with open("dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    # Простой и четкий формат
                    text = f"Вопрос: {item['instruction']}\nОтвет: {item['output']}<|endoftext|>"
                    data.append({"text": text})

        print(f"✅ Загружено {len(data)} примеров")
        return Dataset.from_list(data)

    def setup_model_and_tokenizer(self):
        """Настраивает модель правильно"""
        print("🧠 Настраиваем исправленную модель...")

        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Простая настройка токенизатора
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=None
        )

        # Простая LoRA конфигурация
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],
            bias="none",
            inference_mode=False,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print("✅ Модель настроена!")

    def tokenize_function(self, examples):
        """Правильная токенизация"""
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Убираем padding здесь
            max_length=256,
            return_tensors=None  # Возвращаем списки, не тензоры
        )

        # Копируем input_ids в labels для обучения языковой модели
        result["labels"] = result["input_ids"].copy()

        return result

    def train(self):
        """Исправленное обучение"""
        print("🚀 Начинаем исправленное обучение...")

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

        # Исправленные настройки обучения
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=10,
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
            weight_decay=0.01,
            fp16=False,
            gradient_checkpointing=False,  # Отключаем для стабильности
            dataloader_num_workers=0,
            report_to=None,  # Отключаем логирование
        )

        # Правильный коллатор данных
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
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
    print("🔧 ИСПРАВЛЕННОЕ ОБУЧЕНИЕ СУПЕР МОДЕЛИ")
    print("=" * 60)

    trainer = FixedSuperTrainer()
    trainer.train()

    print("\n✅ ГОТОВО!")


if __name__ == "__main__":
    main()
