import json
import os

# Путь к вашей модели
MODEL_PATH = "C:/Users/Asem/diplomProject/ai_model_server/model"  # Замените на ваш путь

# Проверяем конфигурационный файл
config_path = os.path.join(MODEL_PATH, "adapter_config.json")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Содержимое adapter_config.json:")
    print(json.dumps(config, indent=2))
    
    # Проверяем наличие проблемного параметра
    if 'corda_config' in config:
        print("\n❌ Найден проблемный параметр 'corda_config'")
        print("Возможно, это должно быть 'lora_config' или параметр нужно удалить")
    else:
        print("\n✅ Параметр 'corda_config' не найден")
else:
    print(f"❌ Файл конфигурации не найден: {config_path}")
