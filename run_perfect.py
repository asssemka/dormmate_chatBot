"""
Запускает идеально точный чатбот
"""
import os
import sys
import subprocess
import time


def check_requirements():
    """Проверяет наличие необходимых библиотек"""
    required_packages = ['fastapi', 'uvicorn', 'transformers', 'torch', 'peft']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Отсутствуют необходимые библиотеки: {', '.join(missing_packages)}")
        print("📦 Установка библиотек...")

        for package in missing_packages:
            subprocess.run([sys.executable, "-m", "pip", "install", package])

        print("✅ Библиотеки установлены")


def check_files():
    """Проверяет наличие необходимых файлов"""
    required_files = ['perfect_database.py', 'super_ai_server.py', 'dataset.jsonl']
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Отсутствуют необходимые файлы: {', '.join(missing_files)}")
        return False

    return True


def run_server():
    """Запускает сервер"""
    print("🚀 Запуск идеально точного чатбота...")

    try:
        subprocess.run([sys.executable, "super_ai_server.py"])
    except KeyboardInterrupt:
        print("\n⛔ Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")


def main():
    """Основная функция"""
    print("🎯 ЗАПУСК ИДЕАЛЬНО ТОЧНОГО ЧАТБОТА")
    print("=" * 60)

    # Проверяем библиотеки
    check_requirements()

    # Проверяем файлы
    if not check_files():
        print("❌ Невозможно запустить чатбот из-за отсутствия необходимых файлов")
        return

    # Запускаем сервер
    run_server()


if __name__ == "__main__":
    main()
