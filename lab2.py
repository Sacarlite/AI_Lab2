import torch
import pandas as pd
import torchaudio
import yt_dlp
import io
import requests
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2CTCTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys

# 1. Подготовка данных
def load_and_filter_dataset(file_path):
    """Загрузка и фильтрация датасета."""
    data = pd.read_csv(file_path)
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

def download_audio_to_memory(youtube_id):
    """Скачивание аудио с YouTube по ID."""
    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
    ydl_opts = {
        'format': 'bestaudio',
        'quiet': True,
        'no_progress': True,
        'logger': None,
        'outtmpl': '-',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            audio_url = info_dict['url']
        response = requests.get(audio_url, timeout=10)
        response.raise_for_status()
        return io.BytesIO(response.content), None
    except Exception as e:
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
                    waveform, sample_rate = torchaudio.load(audio_data)
                    if waveform.shape[1] > 0:
                        self.valid_data.append((yt_id, label))
                except Exception as e:
                    print(f"Ошибка обработки аудио {yt_id}: {str(e)}", file=sys.stderr)
            else:
                print(f"Ошибка загрузки {yt_id}: {error}", file=sys.stderr)
        print(f"Успешно загружено {len(self.valid_data)} аудиофайлов из {len(self.youtube_ids)}.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        yt_id, label = self.valid_data[idx]
        audio_data, error = download_audio_to_memory(yt_id)
        if audio_data:
            waveform, sample_rate = torchaudio.load(audio_data)
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
    youtube_ids = extract_youtube_links(data)

    # Разделение на train/test
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        youtube_ids, data['category'].tolist(), test_size=0.2, random_state=42
    )

    # Попробуем сначала основную модель, затем запасную
    model_name = "facebook/wav2vec2-large-xlsr-53"
    backup_model_name = "facebook/wav2vec2-base-960h"

    try:
        print(f"Загрузка токенизатора для {model_name}...")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        print("Токенизатор успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки токенизатора для {model_name}: {e}")
        print(f"Пробуем запасную модель: {backup_model_name}")
        try:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(backup_model_name)
            model_name = backup_model_name  # Переключаемся на запасную модель
            print(f"Токенизатор для {backup_model_name} успешно загружен.")
        except Exception as e:
            print(f"Ошибка загрузки токенизатора для {backup_model_name}: {e}")
            print("Проверьте интернет-соединение и кэш Hugging Face.")
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

    # Создание датасетов
    train_dataset = AudioDataset(train_ids, train_labels, processor, max_samples=40)
    test_dataset = AudioDataset(test_ids, test_labels, processor, max_samples=10)
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
            print(f"Эпоха {epoch+1}/{num_epochs}, Батч {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch+1}/{num_epochs}, Средний Loss: {avg_loss:.4f}")

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

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

if __name__ == "__main__":
    main()