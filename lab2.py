
import torch
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from AudioDataset import AudioDataset
# Чтение данных из CSV файла
original_file = 'youtube.csv'
data = pd.read_csv(original_file)

# Валидация данных CSV: проверяем, что в данных есть необходимые колонки
required_columns = ["link", "category"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV должен содержать колонки: {required_columns}")

# Маппинг категорий на integer для обучения модели
category_mapping = {label: idx for idx, label in enumerate(data["category"].unique())}
category_count = len(data["category"].unique())  # Количество уникальных категорий
data["category"] = data["category"].map(category_mapping)  # Заменяем строковые категории на числа
print(f"Маппинг категорий: {category_mapping}")

# Конфигурация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Выбираем устройство: GPU или CPU
model_name = "facebook/wav2vec2-large-xlsr-53"  # Используем предобученную модель Wav2Vec2
batch_size = 16  # Размер мини-батча

# Загрузка процессора и модели Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=category_count)
model.to(device)  # Переносим модель на устройство (GPU или CPU)

# Разделение данных на train/test
links = data["link"].astype(str).tolist()  # Ссылки на YouTube
labels = data["category"].astype(int).tolist()  # Метки категорий

# Разделение на тренировочную и тестовую выборки
train_links, test_links, train_labels, test_labels = train_test_split(
    links, labels, test_size=0.2, random_state=42, stratify=labels
)
# Создание DataLoader для тренировочных и тестовых данных
train_dataset = AudioDataset(train_links, train_labels, processor)
test_dataset = AudioDataset(test_links, test_labels, processor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Обучение модели
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


def train_model(model, train_loader, epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)  # Оптимизатор AdamW с маленьким шагом
    loss_fn = CrossEntropyLoss()  # Функция потерь для многоклассовой классификации

    model.train()  # Переводим модель в режим обучения
    for epoch in range(epochs):
        total_loss = 0  # Суммарная потеря за эпоху
        for batch in train_loader:
            input_values, attention_mask, labels = batch  # Разделяем батч на токены и метки
            input_values, attention_mask, labels = input_values.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(input_values=input_values, attention_mask=attention_mask)  # Прогоняем через модель
            loss = loss_fn(outputs.logits, labels)  # Вычисляем потерю
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Обновляем параметры модели
            total_loss += loss.item()  # Добавляем потерю текущего батча
        print(f"Эпоха {epoch + 1}, Потеря: {total_loss:.4f}")  # Выводим потерю за эпоху


# Обучаем модель
train_model(model, train_loader, epochs=3)


# Оценка на тестовой выборке
def evaluate_model(model, test_loader):
    model.eval()  # Переводим модель в режим оценки
    predictions = []  # Список предсказаний
    true_labels = []  # Список истинных меток

    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        for batch in test_loader:
            input_values, attention_mask, labels = batch  # Разделяем батч на токены и метки
            input_values, attention_mask, labels = input_values.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_values=input_values, attention_mask=attention_mask)  # Прогоняем через модель
            logits = outputs.logits  # Получаем логиты (до применения softmax)
            batch_predictions = torch.argmax(logits,
                                             dim=-1).cpu().tolist()  # Получаем предсказания (максимальные логиты)
            predictions.extend(batch_predictions)  # Добавляем предсказания в список
            true_labels.extend(labels.cpu().tolist())  # Добавляем истинные метки в список

    # Вычисление метрик
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=1)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=1)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=1)
    accuracy = accuracy_score(true_labels, predictions)

    # Вывод результатов
    print(f"Точность (Precision): {precision:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"F1-метрика: {f1:.4f}")
    print(f"Точность классификации (Accuracy): {accuracy:.4f}")


# Оценка модели на тестовых данных
evaluate_model(model, test_loader)