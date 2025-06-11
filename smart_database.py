import json
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

class SmartDatabase:
    def __init__(self, data_path: str = "dataset.jsonl"):
        self.qa_pairs = []
        self.synonyms = self._create_synonyms()
        self.slang_mapping = self._create_slang_mapping()
        self.stop_phrases = self._create_stop_phrases()
        self.load_data(data_path)
        self.create_smart_index()

    def _create_synonyms(self) -> Dict[str, List[str]]:
        return {
            'общежитие': [
                'общага', 'общак', 'общежития', 'общаги', 'общагах', 'общаге',
                'дом студентов', 'дс', 'студдом', 'студенческий дом',
                'общежитий', 'общежитие', 'общежитием', 'общежитии',
                'студгородок', 'студенческое общежитие', 'студенческая общага',
                'жилье', 'жильё', 'место', 'комната', 'койка', 'койко-место'
            ],
            'где': ['куда', 'адрес', 'местоположение', 'расположение', 'находится', 'расположен', 'искать', 'найти'],
            'как': ['каким образом', 'способ', 'метод', 'процедура'],
            'подать': ['отправить', 'послать', 'сдать', 'передать', 'подавать', 'оформить'],
            'заявку': ['заявление', 'документы', 'бумаги', 'анкету', 'форму'],
            'документы': ['справки', 'бумаги', 'документация', 'бумажки', 'справочки'],
            'оплатить': ['заплатить', 'внести плату', 'расплатиться', 'платить', 'оплачивать'],
            'отказали': ['не дали', 'отклонили', 'не одобрили', 'не прошел', 'не прошла', 'отказ'],
            'апелляция': ['жалоба', 'пересмотр', 'обжалование', 'повторная подача'],
            'комната': ['комнатка', 'помещение', 'номер', 'блок', 'секция'],
            'человек': ['людей', 'студентов', 'жильцов', 'народу', 'чел'],
            'можно': ['разрешено', 'позволено', 'допустимо', 'возможно'],
            'нельзя': ['запрещено', 'не разрешено', 'не допустимо', 'невозможно']
        }

    def _create_slang_mapping(self) -> Dict[str, str]:
        return {
            'че': 'что',
            'чё': 'что',
            'чо': 'что',
            'шо': 'что',
            'кто': 'кто',
            'када': 'когда',
            'щас': 'сейчас',
            'тока': 'только',
            'токо': 'только',
            'нада': 'надо',
            'надо': 'нужно',
            'канеш': 'конечно',
            'канешн': 'конечно',
            'норм': 'нормально',
            'окей': 'хорошо',
            'ок': 'хорошо',
            'спс': 'спасибо',
            'пжл': 'пожалуйста',
            'плз': 'пожалуйста',
            'инфа': 'информация',
            'инфо': 'информация',

            # Сокращения общежитий
            'дс3': 'общежитие дс3',
            'дс2а': 'общежитие дс2а',
            'дс2б': 'общежитие дс2б',
            'дс-3': 'общежитие дс3',
            'дс-2а': 'общежитие дс2а',
            'дс-2б': 'общежитие дс2б',
            'дс 3': 'общежитие дс3',
            'дс 2а': 'общежитие дс2а',
            'дс 2б': 'общежитие дс2б',

            # Варианты слова "общежитие"
            'общага': 'общежитие',
            'общак': 'общежитие',
            'общаги': 'общежитие',
            'общагах': 'общежитие',
            'общаге': 'общежитие',
            'общагу': 'общежитие',
            'общагой': 'общежитие',
            'общаг': 'общежитие',
            'общежития': 'общежитие',
            'общежитий': 'общежитие',
            'общежитием': 'общежитие',
            'общежитии': 'общежитие',

            # Дом студентов
            'дс': 'общежитие',
            'студдом': 'общежитие',
            'студенческий дом': 'общежитие',
            'дом студентов': 'общежитие',
            'студгородок': 'общежитие',

            # Другие варианты
            'жилье': 'общежитие',
            'жильё': 'общежитие',
            'место': 'общежитие',
            'койка': 'общежитие',
            'койко-место': 'общежитие'
        }

    def _create_stop_phrases(self) -> List[str]:
        return [
            "Извините, я не знаю ответа на этот вопрос.",
            "К сожалению, у меня нет информации по этому вопросу.",
            "Этот вопрос выходит за рамки моих знаний.",
            "Я не могу ответить на этот вопрос.",
            "По этому вопросу лучше обратиться к администрации.",
            "Извините, но я не располагаю такой информацией."
        ]

    def load_data(self, data_path: str):
        print(f"Загружаем умные данные из {data_path}...")

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        self.qa_pairs.append({
                            'question': item['instruction'],
                            'answer': item['output'],
                            'keywords': self._extract_smart_keywords(item['instruction']),
                            'normalized_question': self._normalize_text(item['instruction'])
                        })

            print(f"Загружено {len(self.qa_pairs)} умных вопросов и ответов")
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            self._create_fallback_data()

    def _create_fallback_data(self):
        fallback_data = [
            {
                'question': 'Где находится общежитие ДС3?',
                'answer': 'Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А.',
            },
            {
                'question': 'Где находится общежитие ДС2б?',
                'answer': 'Общежитие ДС2б находится по адресу: г. Алматы, Таугуль 34.',
            },
            {
                'question': 'Где находится общежитие ДС2а?',
                'answer': 'Общежитие ДС2А находится по адресу: г. Алматы, Таугуль 32.',
            },
            {
                'question': 'Где находится общежитие Емен?',
                'answer': 'Общежитие Емен находится по адресу: г. Алматы, мкр №10 26/1.',
            },
            {
                'question': 'Мне не дали общежитие',
                'answer': 'Рекомендую обратиться в деканат или отдел студенческого проживания для консультации и возможного пересмотра заявки.',
            }
        ]

        self.qa_pairs = []
        for item in fallback_data:
            self.qa_pairs.append({
                'question': item['question'],
                'answer': item['answer'],
                'keywords': self._extract_smart_keywords(item['question']),
                'normalized_question': self._normalize_text(item['question'])
            })
        print("Создан базовый набор данных")

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        # Заменяем сленг
        words = text.split()
        normalized_words = []
        for word in words:
            # Убираем знаки препинания
            clean_word = re.sub(r'[^\w]', '', word)
            # Заменяем сленг
            if clean_word in self.slang_mapping:
                normalized_words.append(self.slang_mapping[clean_word])
            else:
                normalized_words.append(clean_word)

        return ' '.join(normalized_words)

    def _extract_smart_keywords(self, text: str) -> List[str]:
        normalized_text = self._normalize_text(text)
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'за', 'от', 'к', 'у', 'о', 'из', 'а', 'но', 'или', 'это', 'то',
                      'как', 'что'}
        words = [word for word in normalized_text.split() if word not in stop_words and len(word) > 1]
        extended_keywords = set(words)
        for word in words:
            for main_word, synonyms in self.synonyms.items():
                if word in synonyms or word == main_word:
                    extended_keywords.add(main_word)
                    extended_keywords.update(synonyms)

        return list(extended_keywords)

    def create_smart_index(self):
        self.keyword_index = {}
        self.question_similarity = {}

        for i, qa_pair in enumerate(self.qa_pairs):
            for keyword in qa_pair['keywords']:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(i)

            self.question_similarity[i] = qa_pair['normalized_question']

        print(f"Создан умный индекс из {len(self.keyword_index)} ключевых слов")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    def _get_stop_phrase(self) -> str:
        import random
        return random.choice(self.stop_phrases)

    def find_smart_match(self, question: str) -> Tuple[str, float]:
        normalized_question = self._normalize_text(question)
        # 1. Проверяем точное совпадение
        for qa_pair in self.qa_pairs:
            if qa_pair['normalized_question'] == normalized_question:
                return qa_pair['answer'], 1.0

        # 2. Проверяем специальные паттерны для общежитий
        answer, confidence = self._check_dormitory_patterns(normalized_question)
        if confidence > 0.8:
            return answer, confidence

        # 3. Поиск по ключевым словам
        keywords = self._extract_smart_keywords(question)
        if not keywords:
            return self._get_stop_phrase(), 0.1
        scores = []
        for i, qa_pair in enumerate(self.qa_pairs):
            matched_keywords = set(keywords) & set(qa_pair['keywords'])
            keyword_score = len(matched_keywords) / max(len(keywords), len(qa_pair['keywords']))
            similarity_score = self._calculate_similarity(normalized_question, qa_pair['normalized_question'])
            total_score = (keyword_score * 0.7) + (similarity_score * 0.3)
            special_keywords = ['дс3', 'дс2а', 'дс2б', 'емен', 'общежитие']
            for keyword in special_keywords:
                if keyword in matched_keywords:
                    total_score += 0.2
            scores.append((i, total_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        if scores and scores[0][1] > 0.4:
            best_match_idx = scores[0][0]
            return self.qa_pairs[best_match_idx]['answer'], scores[0][1]
        return self._get_stop_phrase(), 0.1

    def _check_dormitory_patterns(self, normalized_question: str) -> Tuple[str, float]:
        """Проверяет специальные паттерны для общежитий"""
        # Паттерны для ДС3
        ds3_patterns = ['дс3', 'дс 3', 'дс-3', 'общежитие 3', 'третье общежитие', 'третья общага', 'третий дс']
        if any(pattern in normalized_question for pattern in ds3_patterns):
            if any(word in normalized_question for word in
                   ['где', 'адрес', 'находится', 'расположен', 'куда', 'искать', 'найти']):
                return 'Общежитие №3 находится по адресу: г. Алматы, мкр №1 81А.', 0.95

        # Паттерны для ДС2а
        ds2a_patterns = ['дс2а', 'дс 2а', 'дс-2а', 'общежитие 2а', 'вторая а общага', 'дс2 а']
        if any(pattern in normalized_question for pattern in ds2a_patterns):
            if any(word in normalized_question for word in
                   ['где', 'адрес', 'находится', 'расположен', 'куда', 'искать', 'найти']):
                return 'Общежитие ДС2А находится по адресу: г. Алматы, Таугуль 32.', 0.95

        # Паттерны для ДС2б
        ds2b_patterns = ['дс2б', 'дс 2б', 'дс-2б', 'общежитие 2б', 'вторая б общага', 'дс2 б']
        if any(pattern in normalized_question for pattern in ds2b_patterns):
            if any(word in normalized_question for word in
                   ['где', 'адрес', 'находится', 'расположен', 'куда', 'искать', 'найти']):
                return 'Общежитие ДС2б находится по адресу: г. Алматы, Таугуль 34.', 0.95

        # Паттерны для Емен
        emen_patterns = ['емен', 'общежитие емен', 'общага емен']
        if any(pattern in normalized_question for pattern in emen_patterns):
            if any(word in normalized_question for word in
                   ['где', 'адрес', 'находится', 'расположен', 'куда', 'искать', 'найти']):
                return 'Общежитие Емен находится по адресу: г. Алматы, мкр №10 26/1.', 0.95

        # Паттерны для отказа в общежитии
        rejection_patterns = ['не дали', 'отказали', 'не одобрили', 'не прошел', 'не прошла', 'отказ']
        housing_patterns = ['общежитие', 'общага', 'общак', 'дс', 'жилье', 'место']
        if any(pattern in normalized_question for pattern in rejection_patterns):
            if any(pattern in normalized_question for pattern in housing_patterns):
                return 'Рекомендую обратиться в деканат или отдел студенческого проживания для консультации и возможного пересмотра заявки.', 0.9

        # Паттерны для подачи заявки
        application_patterns = ['как подать', 'подача заявки', 'подать заявление', 'оформить заявку',
                                'подавать документы']
        if any(pattern in normalized_question for pattern in application_patterns):
            if any(pattern in normalized_question for pattern in housing_patterns):
                return 'Подать заявку можно через личный кабинет на сайте, заполнив анкету и приложив необходимые документы.', 0.9

        general_housing_questions = [
            ('какие общаги',
             'В университете есть общежития: ДС3 (мкр №1 81А), ДС2А (Таугуль 32), ДС2б (Таугуль 34), Емен (мкр №10 26/1).'),
            ('какие общежития',
             'В университете есть общежития: ДС3 (мкр №1 81А), ДС2А (Таугуль 32), ДС2б (Таугуль 34), Емен (мкр №10 26/1).'),
            ('список общаг',
             'В университете есть общежития: ДС3 (мкр №1 81А), ДС2А (Таугуль 32), ДС2б (Таугуль 34), Емен (мкр №10 26/1).'),
            ('сколько общаг', 'В университете 4 основных общежития: ДС3, ДС2А, ДС2б и Емен.'),
            ('все общежития',
             'В университете есть общежития: ДС3 (мкр №1 81А), ДС2А (Таугуль 32), ДС2б (Таугуль 34), Емен (мкр №10 26/1).')
        ]
        for pattern, answer in general_housing_questions:
            if pattern in normalized_question:
                return answer, 0.9
        return "", 0.0

    def get_smart_answer(self, question: str) -> Tuple[str, float]:
        if not question.strip():
            return self._get_stop_phrase(), 0.1
        if len(question.strip()) < 3:
            return self._get_stop_phrase(), 0.1
        answer, confidence = self.find_smart_match(question)
        if confidence < 0.3:
            return self._get_stop_phrase(), 0.1

        return answer, confidence


if __name__ == "__main__":
    db = SmartDatabase()

    test_questions = [
        "Где ДС3?",
        "Че за адрес у дс2а?",
        "Куда ехать если дали место в дс3?",
        "Мне не дали общагу, че делать?",
        "Как подать заявку?",
        "Какие общаги есть?",
        "Сколько мест в комнатах?",
        "Че можно взять с собой?",
        "Привет как дела?",
        "Где библиотека?",
        "Какая погода?"
    ]

    print("\n ТЕСТИРОВАНИЕ УМНОЙ БАЗЫ ДАННЫХ:")
    for question in test_questions:
        answer, confidence = db.get_smart_answer(question)
        print(f"Вопрос:  {question}")
        print(f"Ответ:  {answer}")
        print(f" Уверенность: {confidence:.2f}")
        print("-" * 50)