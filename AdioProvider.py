
import yt_dlp
import io
import requests


def download_video_to_memory(youtube_url):
    """
    Скачивает видео с YouTube в память, добавлена проверка на доступность видео.

    :param youtube_url: URL видео на YouTube
    :return: поток в памяти с данными видео или None в случае ошибки
    """
    # Настройки для скачивания видео в память
    ydl_opts = {
        'format': 'best',  # Скачиваем лучшее доступное качество
        'quiet': True,  # Отключаем вывод информации
        'extractaudio': False,  # Отключаем извлечение только аудио
        'outtmpl': '-',  # Скачиваем в stdout (то есть в память)
    }

    try:
        # Скачиваем видео с помощью yt-dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Попытка извлечь информацию о видео
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_url = info_dict['url']  # Получаем прямую ссылку на видео

        # Скачиваем видео в память
        response = requests.get(video_url)
        response.raise_for_status()  # Если запрос не успешен, возбуждается исключение
        video_data = io.BytesIO(response.content)
        return video_data

    except yt_dlp.utils.DownloadError as e:
        # Ошибка при скачивании видео (например, видео может быть удалено или приватное)
        print(f"Ошибка скачивания видео: {str(e)}")
    except requests.exceptions.RequestException as e:
        # Ошибка при загрузке видео через HTTP (например, недоступность видео)
        print(f"Ошибка при загрузке видео: {str(e)}")
    except Exception as e:
        # Ловим другие возможные ошибки
        print(f"Произошла ошибка: {str(e)}")

    # Если видео не удалось скачать или произошла ошибка, возвращаем None
    return None
