import os
import shutil

# Файлы, которые можно удалить для очистки проекта
files_to_remove = [
    "improved_server.py",  # Дублирует local_model_server.py
    "inference.py",        # Если не используется
    "hh.py",              # Неясное назначение
    ".pytest_cache"       # Кэш тестов
]

# Папка проекта
project_path = "../../Downloads/cleanup-project"

print("🧹 Очистка проекта...")

for item in files_to_remove:
    item_path = os.path.join(project_path, item)
    
    if os.path.exists(item_path):
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"🗑️ Удален файл: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"🗑️ Удалена папка: {item}")
        except Exception as e:
            print(f"❌ Не удалось удалить {item}: {e}")
    else:
        print(f"⚠️ Не найден: {item}")

print("\n✅ Очистка завершена!")
print("\nОставшиеся важные файлы:")
important_files = [
    "local_model_server.py",
    "test_client.py", 
    "adapter.py",
    "train_and_save_lora.py",
    "dataset.json",
    "requirements.txt"
]

for file in important_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - НЕ НАЙДЕН!")
