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
    # Брать топ n по трешхолду и сделать это n как параметр

    def __init__(
        self,
        document_name: str,
        request: str,
        window_size=1,
        step_size=1,
        tresh_hold=0.78,
        auth=config.auth,
        model=SentenceTransformer("intfloat/multilingual-e5-large"),
    ):
        self.document_name = document_name
        self.request = request
        self.window_size = window_size
        self.step_size = step_size
        self.tresh_hold = tresh_hold
        self.auth = auth
        self.model = model

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

        return new_txt_list
        # доделать ширину окна и сдвиг(плохо работает на шаге (embedding))

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

    def embeding(self, text_query, text_links, tresh_hold):
        """
        Отбор предложений-кандидатов

        Параметры:
        - text_query (str): запрос (текст для валидации)
        - text_links (list): список предложений из источника

        Возвращает:
        - словарь с парой ключ-знанение, где ключ - порядновый номер предложения-кандидата
                значние - само предложение-кандидат
        """

        list_of_candidates = []
        dict_of_all_candidats = {}

        text_links.append(text_query)

        # embeddings_query = model.encode(text_query, normalize_embeddings=False)
        embeddings_links = self.model.encode(text_links, normalize_embeddings=False)
        embeddings_query = embeddings_links[-1]
        embeddings_links = embeddings_links[:-1]

        answer = util.cos_sim(embeddings_query, embeddings_links)[0]

        text_links.pop()

        for i in range(len(answer)):
            dict_of_all_candidats[i] = {
                "text": text_links[i],
                "embedder_score": float(answer[i]),
            }
            if answer[i] > self.tresh_hold:
                list_of_candidates.append(i)

        return dict_of_all_candidats, list_of_candidates

    def get_token(self, auth_token, scope="GIGACHAT_API_PERS"):
        """
        Выполняет POST-запрос к эндпоинту, который выдает токен.

        Параметры:
        - auth_token (str): токен авторизации, необходимый для запроса.
        - область (str): область действия запроса API. По умолчанию — «GIGACHAT_API_PERS».

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
            "Authorization": f"Basic {auth_token}",
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

    def get_chat_completion(self, auth_token, user_message):
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

    def answer(self, text_query, links, tresh_hold):

        text_links_prep = links
        text_links, list_cand = self.embeding(text_query, text_links_prep, tresh_hold)

        response = self.get_token(self.auth)
        if response != -1:
            giga_token = response.json()["access_token"]

        answer = []

        for i in text_links:
            text_n = text_links[i]["text"]

            if i in list_cand:
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
                elif "нет" in llm_response:
                    answer.append(
                        {
                            "sentence_idx": i,
                            "text": text_n,
                            "color": "red",
                            "embedder_score": text_links[i]["embedder_score"],
                            "LLM_response": llm_response,
                        }
                    )

        return answer

    def get_modified_file(self):
        request = self.request
        path = f"files/{self.document_name}"
        document = fitz.open(path)
        text = self.prepare_text(document)
        sentences = self.answer(request, text, self.tresh_hold)
        self.modifi_document(sentences, document)

