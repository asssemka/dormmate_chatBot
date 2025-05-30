"""
Создает точную базу данных для поиска ответов
"""
import json
import re
from typing import Dict, List, Tuple, Optional


class PerfectDatabase:
    def __init__(self, data_path: str = "dataset.jsonl"):
        """Инициализирует базу данных с точными ответами"""
        self.qa_pairs = []
        self.load_data(data_path)
        self.create_index()

    def load_data(self, data_path: str):
        """Загружает данные из JSONL файла"""
        print(f"📚 Загружаем точные данные из {data_path}...")

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        self.qa_pairs.append({
                            'question': item['instruction'],
                            'answer': item['output'],
                            'keywords': self._extract_keywords(item['instruction'])
                        })

            print(f"✅ Загружено {len(self.qa_pairs)} вопросов и ответов")
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            # Создаем минимальный набор данных
            self._create_fallback_data()

    def _create_fallback_data(self):
        """Создает минимальный набор данных в случае ошибки"""
        self.qa_pairs = [
            {
                'question': 'Где находится общежитие ДС3?',
                'answer': 'Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А.',
                'keywords': ['дс3', 'общежитие', 'где', 'находится']
            },
            {
                'question': 'Где находится общежитие ДС2б?',
                'answer': 'Общежитие ДС2б находится по адресу: г. Алматы, Таугуль 34.',
                'keywords': ['дс2б', 'общежитие', 'где', 'находится']
            },
            {
                'question': 'Где находится общежитие ДС2а?',
                'answer': 'Общежитие ДС2А находится по адресу: г. Алматы, Таугуль 32.',
                'keywords': ['дс2а', 'общежитие', 'где', 'находится']
            },
            {
                'question': 'Где находится общежитие Емен?',
                'answer': 'Общежитие Емен находится по адресу: г. Алматы, мкр №10 26/1.',
                'keywords': ['емен', 'общежитие', 'где', 'находится']
            },
            {
                'question': 'Мне не дали общежитие',
                'answer': 'Рекомендую обратиться в деканат или отдел студенческого проживания для консультации и возможного пересмотра заявки.',
                'keywords': ['не', 'дали', 'общежитие']
            }
        ]
        print("⚠️ Создан минимальный набор данных")

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые слова из текста"""
        # Приводим к нижнему регистру и удаляем знаки препинания
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # Удаляем стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'за', 'от', 'к', 'у', 'о', 'из', 'что', 'как', 'а', 'но', 'или'}
        words = [word for word in text.split() if word not in stop_words]

        # Добавляем специальные ключевые слова для общежитий
        special_keywords = ['дс3', 'дс2а', 'дс2б', 'емен', 'общежитие', 'адрес']
        for keyword in special_keywords:
            if keyword in text.lower() and keyword not in words:
                words.append(keyword)

        return words

    def create_index(self):
        """Создает индекс для быстрого поиска"""
        self.keyword_index = {}

        for i, qa_pair in enumerate(self.qa_pairs):
            for keyword in qa_pair['keywords']:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(i)

        print(f"✅ Создан индекс из {len(self.keyword_index)} ключевых слов")

    def find_exact_match(self, question: str) -> Optional[str]:
        """Ищет точное совпадение вопроса"""
        question_lower = question.lower().strip()

        for qa_pair in self.qa_pairs:
            if qa_pair['question'].lower().strip() == question_lower:
                return qa_pair['answer']

        return None

    def find_best_match(self, question: str) -> Tuple[str, float]:
        """Находит наилучшее совпадение для вопроса"""
        # Сначала проверяем точное совпадение
        exact_match = self.find_exact_match(question)
        if exact_match:
            return exact_match, 1.0

        # Извлекаем ключевые слова из вопроса
        keywords = self._extract_keywords(question)

        # Если нет ключевых слов, возвращаем общий ответ
        if not keywords:
            return "Извините, я не понимаю ваш вопрос. Пожалуйста, уточните его.", 0.3

        # Считаем совпадения для каждого вопроса в базе
        scores = []
        for i, qa_pair in enumerate(self.qa_pairs):
            matched_keywords = set(keywords) & set(qa_pair['keywords'])
            score = len(matched_keywords) / max(len(keywords), len(qa_pair['keywords']))

            # Бонус за совпадение специальных ключевых слов
            for keyword in ['дс3', 'дс2а', 'дс2б', 'емен']:
                if keyword in matched_keywords:
                    score += 0.3

            scores.append((i, score))

        # Сортируем по убыванию оценки
        scores.sort(key=lambda x: x[1], reverse=True)

        # Если лучшая оценка выше порога, возвращаем соответствующий ответ
        if scores and scores[0][1] > 0.4:
            best_match_idx = scores[0][0]
            return self.qa_pairs[best_match_idx]['answer'], scores[0][1]

        # Иначе возвращаем общий ответ
        return "Извините, я не нашел точного ответа на ваш вопрос. Пожалуйста, уточните его.", 0.2

    def get_answer(self, question: str) -> Tuple[str, float]:
        """Получает ответ на вопрос с оценкой уверенности"""
        # Проверяем специальные случаи для общежитий
        question_lower = question.lower()

        # Проверяем на вопросы об адресах общежитий
        if 'дс3' in question_lower or 'дс 3' in question_lower or 'дс-3' in question_lower:
            if any(word in question_lower for word in ['где', 'адрес', 'находится', 'расположен']):
                return 'Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А.', 1.0

        if 'дс2а' in question_lower or 'дс 2а' in question_lower or 'дс-2а' in question_lower:
            if any(word in question_lower for word in ['где', 'адрес', 'находится', 'расположен']):
                return 'Общежитие ДС2А находится по адресу: г. Алматы, Таугуль 32.', 1.0

        if 'дс2б' in question_lower or 'дс 2б' in question_lower or 'дс-2б' in question_lower:
            if any(word in question_lower for word in ['где', 'адрес', 'находится', 'расположен']):
                return 'Общежитие ДС2б находится по адресу: г. Алматы, Таугуль 34.', 1.0

        if 'емен' in question_lower:
            if any(word in question_lower for word in ['где', 'адрес', 'находится', 'расположен']):
                return 'Общежитие Емен находится по адресу: г. Алматы, мкр №10 26/1.', 1.0

        # Проверяем на вопросы о неполучении общежития
        if ('не дали' in question_lower or 'отказали' in question_lower) and 'общежити' in question_lower:
            return 'Рекомендую обратиться в деканат или отдел студенческого проживания для консультации и возможного пересмотра заявки.', 1.0

        # Для остальных вопросов используем поиск по базе
        return self.find_best_match(question)


# Тестирование базы данных
if __name__ == "__main__":
    db = PerfectDatabase()

    test_questions = [
        "Где находится общежитие ДС3?",
        "Адрес ДС2а?",
        "Как найти общежитие Емен?",
        "Мне не дали общежитие, что делать?",
        "Какие документы нужны для общежития?",
        "Сколько человек живёт в комнате?",
    ]

    print("\n🧪 ТЕСТИРОВАНИЕ БАЗЫ ДАННЫХ:")
    for question in test_questions:
        answer, confidence = db.get_answer(question)
        print(f"❓ {question}")
        print(f"🤖 {answer}")
        print(f"📊 Уверенность: {confidence:.2f}")
        print("-" * 50)
