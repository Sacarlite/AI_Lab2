import torch
import pandas as pd
import torchaudio
import yt_dlp
import io
import requests
import os
import shutil
import gc
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2CTCTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys


# 1. Подготовка данных
def load_and_filter_dataset(file_path):
    """Загрузка и фильтрация датасета с обработкой кодировки."""
    encodings = ['utf-8', 'cp1251', 'latin1', 'utf-8-sig']
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Файл успешно прочитан с кодировкой: {encoding}")
            break
        except UnicodeDecodeError as e:
            print(f"Ошибка декодирования с {encoding}: {e}")
            continue
    else:
        raise ValueError("Не удалось прочитать файл с известными кодировками. Проверьте файл на повреждения.")

    data = data.dropna(subset=['link', 'category'])
    category_mapping = {label: idx for idx, label in enumerate(data['category'].unique())}
    data['category'] = data['category'].map(category_mapping)
    print(f"Категории: {category_mapping}")
    return data, category_mapping


def extract_youtube_links(dataframe, column_name='link'):
    """Извлечение ссылок из датасета."""
    if column_name not in dataframe.columns:
        raise ValueError(f"Столбец '{column_name}' не найден.")
    return dataframe[column_name].dropna().tolist()


def download_audio_to_memory(youtube_id, cookies_path=None):
    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
    ydl_opts = {
        'format': 'bestaudio',
        'quiet': True,  # Без лишних логов
        'no_progress': True,  # Без прогресса
        'outtmpl': '-',  # Загружаем в stdout
    }

    if cookies_path:
        ydl_opts['cookies'] = cookies_path  # Передаем cookies для обхода CAPTCHA

    try:
        # Скачивание аудио с помощью yt-dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            audio_url = info_dict['url']

        # Загружаем аудио с полученной ссылки
        response = requests.get(audio_url, timeout=10)
        response.raise_for_status()  # Проверка на успешность загрузки
        print(f"Аудио для {youtube_id} успешно скачано.")
        return io.BytesIO(response.content), None

    except yt_dlp.utils.DownloadError as e:
        print(f"Ошибка скачивания с YouTube (yt-dlp): {e}")
        return None, str(e)

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при скачивании аудио: {e}")
        return None, str(e)

    except Exception as e:
        print(f"Неизвестная ошибка при скачивании: {e}")
        return None, str(e)


# 2. Кастомный Dataset
class AudioDataset(Dataset):
    def __init__(self, youtube_ids, labels, processor, max_samples=None):
        self.youtube_ids = youtube_ids[:max_samples] if max_samples else youtube_ids
        self.labels = labels[:max_samples] if max_samples else labels
        self.processor = processor
        self.valid_data = []

        for idx, (yt_id, label) in enumerate(zip(self.youtube_ids, self.labels)):
            audio_data, error = download_audio_to_memory(yt_id)
            if audio_data:
                try:
                    waveform, sample_rate = torchaudio.load(audio_data, format='m4a')
                    if waveform.shape[1] > 0:
                        self.valid_data.append((yt_id, label))
                except Exception as e:
                    print(f"Ошибка обработки аудио {yt_id}: {str(e)}", file=sys.stderr)
            else:
                print(f"Пропуск {yt_id} из-за ошибки загрузки.", file=sys.stderr)
        print(f"Успешно загружено {len(self.valid_data)} аудиофайлов из {len(self.youtube_ids)}.")
        if len(self.valid_data) == 0:
            raise ValueError(
                "Не удалось загрузить ни одного аудиофайла. Проверьте доступность видео или формат данных.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        yt_id, label = self.valid_data[idx]
        audio_data, error = download_audio_to_memory(yt_id)
        if audio_data:
            waveform, sample_rate = torchaudio.load(audio_data, format='m4a')
            inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            return inputs.input_values.squeeze(), inputs.attention_mask.squeeze(), torch.tensor(label)
        else:
            raise RuntimeError(f"Не удалось загрузить аудио для {yt_id}: {error}")


# 3. Основная функция
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Загрузка и фильтрация данных
    data, category_mapping = load_and_filter_dataset('youtube.csv')

    # Извлечение и вывод ссылок
    youtube_ids = extract_youtube_links(data)
    print("\nИзвлечённые ссылки на видео:")
    for idx, yt_id in enumerate(youtube_ids, 1):
        print(f"{idx}. {yt_id}")
    print(f"Всего ссылок: {len(youtube_ids)}")

    # Запрос у пользователя количества аудио для train и test
    while True:
        try:
            train_samples = int(
                input("\nСколько аудиофайлов скачать для тренировочного набора? (макс. 80% от общего): "))
            test_samples = int(input("Сколько аудиофайлов скачать для тестового набора? (макс. 20% от общего): "))
            total_samples = train_samples + test_samples
            max_total = len(youtube_ids)
            if total_samples <= 0 or total_samples > max_total:
                print(f"Ошибка: общее количество должно быть от 1 до {max_total}.")
            elif train_samples < 0 or test_samples < 0:
                print("Ошибка: количество не может быть отрицательным.")
            else:
                break
        except ValueError:
            print("Ошибка: введите целое число.")

    # Разделение на train/test с учётом пользовательского ввода
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        youtube_ids, data['category'].tolist(), test_size=test_samples / total_samples,
        train_size=train_samples / total_samples, random_state=42
    )

    # Загрузка процессора и модели
    model_name = "facebook/wav2vec2-large-xlsr-53"
    backup_model_name = "facebook/wav2vec2-base-960h"

    try:
        print(f"\nЗагрузка токенизатора для {model_name}...")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        print("Токенизатор успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки токенизатора для {model_name}: {e}")
        print(f"Пробуем запасную модель: {backup_model_name}")
        try:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(backup_model_name)
            model_name = backup_model_name
            print(f"Токенизатор для {backup_model_name} успешно загружен.")
        except Exception as e:
            print(f"Ошибка загрузки токенизатора для {backup_model_name}: {e}")
            return

    try:
        print(f"Загрузка процессора для {model_name}...")
        processor = Wav2Vec2Processor.from_pretrained(model_name, tokenizer=tokenizer)
        print("Процессор успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки процессора: {e}")
        return

    try:
        print(f"Загрузка модели {model_name}...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, num_labels=len(category_mapping)
        )
        model.to(device)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # Создание датасетов с загрузкой аудио
    print("\nСоздание тренировочного датасета:")
    train_dataset = AudioDataset(train_ids, train_labels, processor, max_samples=train_samples)
    print("\nСоздание тестового датасета:")
    test_dataset = AudioDataset(test_ids, test_labels, processor, max_samples=test_samples)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    # Обучение модели
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_values, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            print(f"Эпоха {epoch + 1}/{num_epochs}, Батч {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch + 1}/{num_epochs}, Средний Loss: {avg_loss:.4f}")

    # Оценка модели
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_values, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_values, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # Метрики
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
    recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # Очистка памяти
    print("\nОчистка памяти...")
    del model, train_dataset, test_dataset, train_loader, test_loader
    gc.collect()


if __name__ == "__main__":
    main()