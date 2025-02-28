# Функция для извлечения ссылок на видео из датасета
def extract_youtube_links(dataframe, column_name='link'):

    # Проверяем наличие столбца с ссылками
    if column_name not in dataframe.columns:
        raise ValueError(f"Столбец '{column_name}' не найден в датасете.")

    # Извлекаем ссылки и возвращаем их как список
    youtube_links = dataframe[column_name].dropna().tolist()
    return youtube_links
