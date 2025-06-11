from flask import Flask, request, jsonify, render_template
from smart_database import SmartDatabase
from smart_translate import translate_text   # добавляем импорт

app = Flask(__name__)
database = SmartDatabase()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    lang = data.get('lang', 'KZ').upper()  # по умолчанию русский

    # 1. Переводим вопрос на русский (если нужно)
    if lang != "RU":
        try:
            question_ru = translate_text(question, lang, "RU")
        except Exception as e:
            return jsonify({'error': f'Ошибка перевода вопроса: {str(e)}'}), 502
    else:
        question_ru = question

    # 2. Получаем ответ на русском
    answer, confidence = database.get_smart_answer(question_ru)

    # 3. Переводим ответ обратно на язык пользователя (если нужно)
    if lang != "RU":
        try:
            answer_translated = translate_text(answer, "RU", lang)
        except Exception as e:
            return jsonify({'error': f'Ошибка перевода ответа: {str(e)}'}), 502
    else:
        answer_translated = answer

    return jsonify({
        'answer': answer_translated,
        'confidence': confidence,
        'model_info': 'smart_database',
        'generation_time': 0.1,
        'source': 'database'
    })

if __name__ == '__main__':
    app.run(debug=True)
