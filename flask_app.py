from flask import Flask, request, jsonify, render_template
from smart_database import SmartDatabase

app = Flask(__name__)

# Инициализируем базу данных (без модели для экономии ресурсов)
database = SmartDatabase()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')

    # Получаем ответ из базы данных
    answer, confidence = database.get_smart_answer(question)

    return jsonify({
        'answer': answer,
        'confidence': confidence,
        'model_info': 'smart_database',
        'generation_time': 0.1,
        'source': 'database'
    })


if __name__ == '__main__':
    app.run(debug=True)