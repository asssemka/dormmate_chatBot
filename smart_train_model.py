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

class SmartTrainer:
    def __init__(self):
        self.model_name = "ai-forever/rugpt3small_based_on_gpt2"
        self.output_dir = "./smart_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_enhanced_dataset(self):
        print("Создаем расширенный датасет...")
        base_data = []
        try:
            with open("dataset.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        base_data.append(item)
        except:
            base_data = [
                {"instruction": "Где находится общежитие ДС3?",
                 "output": "Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А."},
                {"instruction": "Где находится общежитие ДС2а?",
                 "output": "Общежитие ДС2А находится по адресу: г. Алматы, Таугуль 32."},
                {"instruction": "Где находится общежитие ДС2б?",
                 "output": "Общежитие ДС2б находится по адресу: г. Алматы, Таугуль 34."},
                {"instruction": "Где находится общежитие Емен?",
                 "output": "Общежитие Емен находится по адресу: г. Алматы, мкр №10 26/1."},
                {"instruction": "Мне не дали общежитие",
                 "output": "Рекомендую обратиться в деканат или отдел студенческого проживания для консультации и возможного пересмотра заявки."}
            ]
        enhanced_data = []
        for item in base_data:
            enhanced_data.append(item)
        slang_variants = {
            "Где находится общежитие ДС3?": [
                "Где ДС3?", "Че за адрес у ДС3?", "Куда ехать если дали ДС3?",
                "ДС3 где находится?", "Адрес ДС3 какой?", "Где третье общежитие?",
                "Куда идти если заселили в ДС3?", "Где третья общага?",
                "Адрес третьей общаги?", "Где дс 3?", "Где дс-3?",
                "Куда ехать в третью общагу?", "Третий дс где?"
            ],
            "Где находится общежитие ДС2а?": [
                "Где ДС2а?", "Че за адрес у ДС2а?", "ДС2а где?",
                "Адрес ДС2а?", "Куда ехать в ДС2а?", "Где дс 2а?",
                "Где дс-2а?", "Вторая а общага где?", "Адрес второй а общаги?"
            ],
            "Где находится общежитие ДС2б?": [
                "Где ДС2б?", "Че за адрес у ДС2б?", "ДС2б где находится?",
                "Адрес ДС2б какой?", "Где дс 2б?", "Где дс-2б?",
                "Вторая б общага где?", "Адрес второй б общаги?"
            ],
            "Где находится общежитие Емен?": [
                "Где Емен?", "Емен где находится?", "Адрес Емена?",
                "Куда ехать в Емен?", "Общага Емен где?", "Емен адрес какой?"
            ],
            "Мне не дали общежитие": [
                "Не дали общагу", "Отказали в общежитии", "Не прошел в общагу",
                "Мне отказали в общежитии", "Че делать если не дали общагу?",
                "Не дали место в общаге", "Отказ в общежитии", "Не прошла в общагу",
                "Не одобрили общежитие", "Отклонили заявку на общагу"
            ],
            "Как подать заявку на общежитие?": [
                "Как подать заявку?", "Че нужно для заявки?", "Как подавать на общагу?",
                "Куда подавать заявку?", "Как оформить заявку на общежитие?",
                "Процедура подачи заявки", "Как подать документы на общагу?",
                "Заявка на общежитие как?", "Как попасть в общагу?"
            ],
            "Какие документы нужны для общежития?": [
                "Че нужно для общаги?", "Какие документы нужны?", "Что взять для общежития?",
                "Какие справки нужны?", "Документы для общаги", "Что нужно для поступления в общагу?",
                "Список документов для общежития", "Бумаги для общаги какие?"
            ],
            "Какие общежития есть в университете?": [
                "Какие общаги есть?", "Список общаг", "Все общежития",
                "Сколько общаг?", "Какие есть общежития?", "Общаги какие есть?",
                "Все общаги университета", "Список всех общежитий"
            ]
        }
        for original_question, variants in slang_variants.items():
            original_answer = None
            for item in base_data:
                if item['instruction'] == original_question:
                    original_answer = item['output']
                    break

            if original_answer:
                for variant in variants:
                    enhanced_data.append({
                        "instruction": variant,
                        "output": original_answer
                    })

        unknown_questions = [
            "Привет как дела?",
            "Какая погода?",
            "Где библиотека?",
            "Когда каникулы?",
            "Что на обед?",
            "Как дела?",
            "Че происходит?",
            "Расскажи анекдот",
            "Сколько времени?"
        ]

        stop_answers = [
            "Извините, я не знаю ответа на этот вопрос.",
            "К сожалению, у меня нет информации по этому вопросу.",
            "Этот вопрос выходит за рамки моих знаний.",
            "По этому вопросу лучше обратиться к администрации."
        ]

        for i, question in enumerate(unknown_questions):
            enhanced_data.append({
                "instruction": question,
                "output": stop_answers[i % len(stop_answers)]
            })

        print(f"Создан расширенный датасет: {len(enhanced_data)} примеров")
        return enhanced_data

    def prepare_training_data(self, enhanced_data):
        training_texts = []
        for item in enhanced_data:
            text = f"Вопрос: {item['instruction']}\nОтвет: {item['output']}<|endoftext|>"
            training_texts.append({"text": text})
        return Dataset.from_list(training_texts)

    def setup_model_and_tokenizer(self):
        print("Настраиваем умную модель...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
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
        print("Умная модель настроена!")

    def tokenize_function(self, examples):
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256,
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self):
        print("Начинаем обучение умной модели...")
        enhanced_data = self.create_enhanced_dataset()
        dataset = self.prepare_training_data(enhanced_data)
        self.setup_model_and_tokenizer()
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=12,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=150,
            logging_steps=10,
            save_steps=200,
            evaluation_strategy="no",
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            learning_rate=3e-4,
            weight_decay=0.01,
            fp16=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            report_to=None,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        print("Запускаем умное обучение...")
        trainer.train()
        print("Сохраняем умную модель...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        print("Умное обучение завершено!")
        print(f"Умная модель сохранена в: {self.output_dir}")

def main():
    print("ОБУЧЕНИЕ УМНОЙ МОДЕЛИ С РАЗГОВОРНЫМ ЯЗЫКОМ")
    print("=" * 70)
    trainer = SmartTrainer()
    trainer.train()
    print("\n УМНАЯ МОДЕЛЬ ГОТОВА!")

if __name__ == "__main__":
    main()
