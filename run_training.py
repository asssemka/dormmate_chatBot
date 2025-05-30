import subprocess
import sys
import os

def install_requirements():
    """Устанавливает зависимости"""
    print("📦 Устанавливаем зависимости...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_training():
    """Запускает обучение"""
    print("🎓 Запускаем обучение модели...")
    subprocess.check_call([sys.executable, "train_model.py"])

def run_server():
    """Запускает сервер"""
    print("🚀 Запускаем AI сервер...")
    subprocess.check_call([sys.executable, "ai_server.py"])

def main():
    print("🤖 ПОЛНЫЙ ЦИКЛ СОЗДАНИЯ AI ЧАТБОТА")
    print("=" * 50)
    
    choice = input("""
Выберите действие:
1. Установить зависимости
2. Обучить модель
3. Запустить сервер
4. Все по порядку (1→2→3)

Ваш выбор (1-4): """)
    
    if choice == "1":
        install_requirements()
    elif choice == "2":
        run_training()
    elif choice == "3":
        run_server()
    elif choice == "4":
        install_requirements()
        run_training()
        run_server()
    else:
        print("❌ Неверный выбор!")

if __name__ == "__main__":
    main()
