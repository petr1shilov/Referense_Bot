import fitz
import re
from sentence_transformers import SentenceTransformer, util
import requests
import uuid
import json
import os.path
import config
import math


class AnswerAPI:
    # можно сделать как __init__ параметр
    WINDOW_SIZE = 1
    STEP_SIZE = 1
    TRESHOLD_GREEN = 0.78
    TRESHOLD_RED = 0.95
    MAX_CANDIDATES = 10

    def __init__(
        self,
        document_name: str,
        request: str,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        treshold_green=TRESHOLD_GREEN,
        max_candidates_green=MAX_CANDIDATES,
        treshold_red=TRESHOLD_RED,
        max_candidates_red=MAX_CANDIDATES,
        auth=config.auth,
        model=SentenceTransformer("intfloat/multilingual-e5-large"),
    ):
        self.document_name = document_name
        self.request = request
        self.window_size = window_size
        self.step_size = step_size
        self.treshold_green = treshold_green
        self.max_candidates_green = max_candidates_green
        self.treshold_red = treshold_red
        self.max_candidates_red = max_candidates_red
        self.auth = auth
        self.model = model

    def change_params(self):
        request_text = re.split(r'\[.+\]', self.request)
        request_text = request_text[1:]
        clean_request_text = [re.sub(r"[\n]", "", i) for i in request_text]

        self.request = str(clean_request_text[0])
        self.window_size = int(clean_request_text[1])
        self.step_size = int(clean_request_text[2])
        self.treshold_green = float(clean_request_text[3])
        self.max_candidates_green = int(clean_request_text[4])
        self.treshold_red = float(clean_request_text[5])
        self.max_candidates_red = int(clean_request_text[6])

    def diplay_params(self):
        print(f'''\tНазвание документа ==> {self.document_name}\n
        Запрос к документу ==> {self.request}\n
        Ширина скользящего окна {self.window_size}\n
        Шаг скользящего окна ==> {self.step_size}\n
        Порог для прямого подтверждения ==> {self.treshold_green}\n
        Порог для отрицаний ==> {self.treshold_red}\n
        Количество кандидатов ==> {self.max_candidates_green}\n
        Ключ модели ==> {self.auth}\n
        Модель ==> {self.model}''')

    def prepare_text(self, document):
        new_txt_list = []

        text = chr(12).join([page.get_text() for page in document])

        text = re.sub(r"(\.\d+,\d+)", ".", text)
        text = re.sub(r"([.]\d+)", ".", text)
        text = re.sub(r"(\.\–?\d+)", ".", text)
        text_new = re.split(r"(?=[.]\s[А-ЯA-Z])", text)

        for i in text_new:
            new_txt_list.append(re.sub(r"([.]\s)", "", i))

        for j in range(len(new_txt_list)):
            new_txt_list[j] = re.sub(r"([-]\n)", "", new_txt_list[j])

        splitted = [new_txt_list[i * self.step_size : i * self.step_size + self.window_size]\
                     for i in range(math.ceil(len(new_txt_list) / self.step_size))]

        splitted_text = ['. '.join(i) for i in splitted]

        return splitted_text

    def modifi_document(self, sentences, document):
        for page in document:
            for sentens in sentences:
                text_instances = page.search_for(sentens["text"])
                highlight = page.add_highlight_annot(text_instances)
                if sentens["color"] == "green":
                    highlight.set_colors(
                        stroke=[0.8, 1, 0.8]
                    )  # light red color (r, g, b)
                else:
                    highlight.set_colors(
                        stroke=[1, 0.8, 0.8]
                    )  # light red color (r, g, b)
                highlight.update()
        path = f"files/{self.document_name[:-4]}_modified.pdf"
        document.save(path)

    def selection_candidates(self, text_query, text_links):
        """
        Отбор предложений-кандидатов

        Параметры:
        - text_query (str): запрос (текст для валидации)
        - text_links (list): список предложений из источника

        Возвращает:
        - словарь с парой ключ-знанение, где ключ - порядновый номер предложения-кандидата
                значние - само предложение-кандидат
        """
        if self.treshold_red > self.treshold_green:
            min_treshold = self.treshold_green
            max_treshold = self.treshold_red
            flag = True
        else: 
            min_treshold = self.treshold_red
            max_treshold = self.treshold_green
            flag = False
            

        list_of_candidates_green = []
        list_of_candidates_red = []
        dict_of_all_candidats = {}

        text_links.append(text_query)

        embeddings_links = self.model.encode(text_links, normalize_embeddings=False)
        embeddings_query = embeddings_links[-1]
        embeddings_links = embeddings_links[:-1]

        answer = util.cos_sim(embeddings_query, embeddings_links)[0]
        sorted_answer = answer.sort(descending=True)[1].tolist()
        text_links.pop()

        for i in sorted_answer:
            if answer[i] > min_treshold:
                dict_of_all_candidats[i] = {
                    "id": i,
                    "text": text_links[i],
                    "embedder_score": float(answer[i]),
                }
            if flag and answer[i] > min_treshold:
                list_of_candidates_green.append(i)
            elif not flag and answer[i] > max_treshold:
                list_of_candidates_green.append(i)

            if flag and answer[i] > max_treshold:
                list_of_candidates_red.append(i)
            if not flag and answer[i] > min_treshold:
                list_of_candidates_red.append(i)

        if len(list_of_candidates_green) < self.max_candidates_green:
            self.max_candidates_green = len(list_of_candidates_green)

        if len(list_of_candidates_red) < self.max_candidates_red:
            self.max_candidates_red = len(list_of_candidates_red)

        return dict_of_all_candidats, list_of_candidates_green, list_of_candidates_red, flag

    def get_token(self, scope="GIGACHAT_API_PERS"):
        """
        Выполняет POST-запрос к эндпоинту, который выдает токен.

        Параметры:
        - auth_token (str): токен авторизации, необходимый для запроса.
        - scope (str): область действия запроса API. По умолчанию — «GIGACHAT_API_PERS».

        Возвращает:
        - ответ API, где токен и срок его "годности".
        """
        # Создадим идентификатор UUID (36 знаков)
        rq_uid = str(uuid.uuid4())

        # API URL
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        # Заголовки
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rq_uid,
            "Authorization": f"Basic {self.auth}",
        }

        # Тело запроса
        payload = {"scope": scope}

        try:
            # Делаем POST запрос с отключенной SSL верификацией
            # (можно скачать сертификаты Минцифры, тогда отключать проверку не надо)
            response = requests.post(url, headers=headers, data=payload, verify=False)
            return response
        except requests.RequestException as e:
            print(f"Ошибка: {str(e)}")
            return -1

    def get_chat_completion(self, auth_token: str, user_message: str):
        """
        Отправляет POST-запрос к API чата для получения ответа от модели GigaChat.

        Параметры:
        - auth_token (str): Токен для авторизации в API.
        - user_message (str): Сообщение от пользователя, для которого нужно получить ответ.

        Возвращает:
        - str: Ответ от API в виде текстовой строки.
        """
        # URL API, к которому мы обращаемся
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

        # Подготовка данных запроса в формате JSON
        payload = json.dumps(
            {
                "model": "GigaChat-Pro",  # Используемая модель
                "messages": [
                    {
                        "role": "user",  # Роль отправителя (пользователь)
                        "content": user_message,  # Содержание сообщения
                    }
                ],
                "temperature": 1,  # Температура генерации
                "top_p": 0.1,  # Параметр top_p для контроля разнообразия ответов
                "n": 1,  # Количество возвращаемых ответов
                "stream": False,  # Потоковая ли передача ответов
                "max_tokens": 512,  # Максимальное количество токенов в ответе
                "repetition_penalty": 1,  # Штраф за повторения
                "update_interval": 0,  # Интервал обновления (для потоковой передачи)
            }
        )

        # Заголовки запроса
        headers = {
            "Content-Type": "application/json",  # Тип содержимого - JSON
            "Accept": "application/json",  # Принимаем ответ в формате JSON
            "Authorization": f"Bearer {auth_token}",  # Токен авторизации
        }

        # Выполнение POST-запроса и возвращение ответа
        try:
            response = requests.request(
                "POST", url, headers=headers, data=payload, verify=False
            )
            return response
        except requests.RequestException as e:
            # Обработка исключения в случае ошибки запроса
            print(f"Произошла ошибка: {str(e)}")
            return -1

    def answer(self, text_query, links):

        text_links_prep = links
        text_links, list_cand_green, list_cand_red, flag  = self.selection_candidates(text_query, text_links_prep)

        response = self.get_token()
        if response != -1:
            giga_token = response.json()["access_token"]

        answer = []

        if flag:
            for i in text_links:
                text_n = text_links[i]["text"]

                if i in list_cand_green[:self.max_candidates_green]:
                    text_for_api = f'Подтверждается ли текст "{text_query}" текстом "{text_n}"\nОтветь только "да" или "нет"'
                    answer_n = self.get_chat_completion(giga_token, text_for_api)
                    llm_response = str(
                        answer_n.json()["choices"][0]["message"]["content"]
                    ).lower()
                    if "да" in llm_response:
                        answer.append(
                            {
                                "sentence_idx": i,
                                "text": text_n,
                                "color": "green",
                                "embedder_score": text_links[i]["embedder_score"],
                                "LLM_response": llm_response,
                            }
                        )    
                    if i in list_cand_red:
                        list_cand_red.remove(i)   
            for i in text_links:
                text_n = text_links[i]["text"]

                if i in list_cand_red[:self.max_candidates_red]:
                    text_for_api = f'Противоречит ли текст "{text_query}" с текстом "{text_n}"\nОтветь только "да" или "нет"'
                    answer_n = self.get_chat_completion(giga_token, text_for_api)
                    llm_response = str(
                        answer_n.json()["choices"][0]["message"]["content"]
                    ).lower()
                    if "да" in llm_response:
                        answer.append(
                            {
                                "sentence_idx": i,
                                "text": text_n,
                                "color": "red",
                                "embedder_score": text_links[i]["embedder_score"],
                                "LLM_response": llm_response,
                            }
                        ) 
        else:
            for i in text_links:
                text_n = text_links[i]["text"]

                if i in list_cand_red[:self.max_candidates_red]:
                    text_for_api = f'Противоречит ли текст "{text_query}" с текстом "{text_n}"\nОтветь только "да" или "нет"'
                    answer_n = self.get_chat_completion(giga_token, text_for_api)
                    llm_response = str(
                        answer_n.json()["choices"][0]["message"]["content"]
                    ).lower()
                    if "да" in llm_response:
                        answer.append(
                            {
                                "sentence_idx": i,
                                "text": text_n,
                                "color": "red",
                                "embedder_score": text_links[i]["embedder_score"],
                                "LLM_response": llm_response,
                            }
                        ) 
                    if i in list_cand_green:
                        list_cand_green.remove(i)
            for i in text_links:
                text_n = text_links[i]["text"]

                if i in list_cand_green[:self.max_candidates_green]:
                    text_for_api = f'Подтверждается ли текст "{text_query}" текстом "{text_n}"\nОтветь только "да" или "нет"'
                    answer_n = self.get_chat_completion(giga_token, text_for_api)
                    llm_response = str(
                        answer_n.json()["choices"][0]["message"]["content"]
                    ).lower()
                    if "да" in llm_response:
                        answer.append(
                            {
                                "sentence_idx": i,
                                "text": text_n,
                                "color": "green",
                                "embedder_score": text_links[i]["embedder_score"],
                                "LLM_response": llm_response,
                            }
                        )                    
        return answer

    def get_modified_file(self):
        try:
            self.change_params()
        except:
            return False
        request = self.request
        path = f"files/{self.document_name}"
        document = fitz.open(path)
        text = self.prepare_text(document)
        sentences = self.answer(request, text)
        self.modifi_document(sentences, document)
        return True






