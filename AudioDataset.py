from torch.utils.data import DataLoader, Dataset
from AdioProvider import download_video_to_memory  # Импортируем функцию для загрузки видео
import torchaudio
import torch

# Создание кастомного Dataset для DataLoader
class AudioDataset(Dataset):
    def __init__(self, youtube_links, labels, processor):
        self.youtube_links = youtube_links
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.youtube_links)

    def __getitem__(self, idx):
        # Скачиваем аудио с YouTube и загружаем его в память
        video_data = download_video_to_memory(self.youtube_links[idx])

        if video_data:
            # Загружаем аудио из памяти
            waveform, sample_rate = torchaudio.load(video_data)
            inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

            # Преобразуем данные в формат для модели
            label = torch.tensor(self.labels[idx])
            return inputs.input_values.squeeze(), inputs.attention_mask.squeeze(), label
        else:
            return None, None, None

