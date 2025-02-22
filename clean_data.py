import pandas as pd
import re


# Функция для очистки данных
def clean_data(original_filepath, cleaned_filepath):
    # Чтение данных из CSV файла
    df = pd.read_csv(original_filepath)

    # Преобразование всех текстовых данных в нижний регистр
    df['title'] = df['title'].apply(lambda x: x.lower())
    df['description'] = df['description'].apply(lambda x: x.lower())

    # Удаление знаков препинания и числовых символов в столбце 'title'
    cleaned_titles = list()
    for title in df['title']:
        temp = ''
        for character in title:
            # Удаляем символы, которые не являются буквами или пробелами
            if character not in '.,|/\-#0123456789:;&%`()\'!@$?[]{}_*-~=':
                temp += character
        cleaned_titles.append(temp)

    df['cleaned_title'] = cleaned_titles

    # Удаление знаков препинания и числовых символов в столбце 'description'
    cleaned_description = list()
    for description in df['description']:
        temp = ''
        for character in description:
            # Удаляем символы, которые не являются буквами или пробелами
            if character not in '.,|/\-#0123456789:;&%`()\'!@$?[]{}_*-~=':
                temp += character
        cleaned_description.append(temp)

    df['description'] = cleaned_description

    # Удаление лишних пробелов между словами
    df['cleaned_title'] = df['cleaned_title'].apply(lambda x: " ".join(x.split()))
    df['description'] = df['description'].apply(lambda x: " ".join(x.split()))

    # Токенизация и удаление стоп-слов
    from nltk import download
    download('punkt_tab')  # Загрузка пунктуационных токенов для токенизации
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Загрузка стоп-слов для английского языка
    stop_words = set(stopwords.words('english'))

    # Добавление специфичных слов, часто встречающихся в описаниях YouTube, в список стоп-слов
    common_words = ['subscribe', 'subscribed', 'subscribers', 'credits', 'find', 'social', 'media',
                    'k', 'notifications', 'notification', 'clicking', 'little', 'bell', 'icon',
                    'show', 'more', 'less', 'instagram', 'twitter', 'facebook', 'gmail',
                    'contact', 'channel', 'isnt', 'share', 'email', '``', "''", '""']
    for c in common_words:
        stop_words.add(c)

    # Пример токенизации для первого описания
    word_tokens = word_tokenize(df['description'][0])

    # Очистка описаний от стоп-слов и лишних ссылок
    cleaned_strings = list()
    for d in df['description']:
        temp = list()
        for word in word_tokenize(d):
            # Удаляем стоп-слова и ссылки (http, gmail, www)
            if (word not in stop_words) and ('http' not in word) and ('gmail' not in word) and ('www' not in word):
                temp.append(word)
        string = " ".join(temp)
        cleaned_strings.append(string)

    df['description'] = cleaned_strings

    # Аналогичная очистка для заголовков видео
    cleaned_strings = list()
    for d in df['cleaned_title']:
        temp = list()
        for word in word_tokenize(d):
            # Удаляем стоп-слова и ссылки (http, gmail, www)
            if (word not in stop_words) and ('http' not in word) and ('gmail' not in word) and ('www' not in word):
                temp.append(word)
        string = " ".join(temp)
        cleaned_strings.append(string)

    df['cleaned_title'] = cleaned_strings

    # Удаление эмодзи
    # Регулярное выражение для поиска и удаления эмодзи
    def remove_emoji(text):
        emoji_pattern = re.compile("["  # регулярное выражение для поиска эмодзи
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    df['cleaned_title'] = df['cleaned_title'].apply(lambda x: remove_emoji(x))
    df['description'] = df['description'].apply(lambda x: remove_emoji(x))

    # Лемматизация — приведение слов к их базовой форме
    # В данном случае используется стемминг с помощью PorterStemmer
    from nltk.stem import PorterStemmer

    porter = PorterStemmer()

    # Применение стемминга к очищенным заголовкам
    df['cleaned_title'] = df['cleaned_title'].apply(lambda x: porter.stem(x))
    df['description'] = df['description'].apply(lambda x: porter.stem(x))

    # Дополнительное применение стемминга по каждому слову в строке
    cleaned_strings = list()
    for d in df['cleaned_title']:
        temp = list()
        for word in d.split():
            temp.append(porter.stem(word))  # Стемминг каждого слова
        string = " ".join(temp)
        cleaned_strings.append(string)

    df['cleaned_title'] = cleaned_strings

    cleaned_strings = list()
    for d in df['description']:
        temp = list()
        for word in d.split():
            temp.append(porter.stem(word))  # Стемминг каждого слова
        string = " ".join(temp)
        cleaned_strings.append(string)

    df['description'] = cleaned_strings

    # Удаление исходного столбца 'title' и сохранение очищенных данных в новый CSV файл
    df.drop('title', axis=1, inplace=True)  # Удаление исходного столбца

    # Сохранение очищенных данных в новый файл
    df.to_csv(cleaned_filepath, index=False)
