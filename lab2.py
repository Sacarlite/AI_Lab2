import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from clean_data import clean_data

# Очищение данных
original_file = 'youtube.csv'
cleaned_file = 'cleaned_data.csv'
clean_data(original_file, cleaned_file)  # Вызов функции очистки данных из clean_data

# Чтение очищенных данных
data = pd.read_csv(cleaned_file)

# Валидация данных CSV: проверяем, что в данных есть необходимые колонки
required_columns = ["link", "description", "category", "cleaned_title"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV должен содержать колонки: {required_columns}")

# Маппинг категорий на integer для обучения модели
category_mapping = {label: idx for idx, label in enumerate(data["category"].unique())}
category_count = len(data["category"].unique())  # Количество уникальных категорий
data["category"] = data["category"].map(category_mapping)  # Заменяем строковые категории на числа
print(f"Маппинг категорий: {category_mapping}")

# Конфигурация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Выбираем устройство: GPU или CPU
model_name = "albert-base-v2"  # Используем предобученную модель ALBERT базового размера
max_length = 128  # Максимальная длина для токенизации
batch_size = 16  # Размер мини-батча

# Загрузка токенизатора и модели ALBERT
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=category_count)
model.to(device)  # Переносим модель на устройство (GPU или CPU)

# Разделение данных на train/test
texts = data["cleaned_title"].astype(str).tolist()  # Тексты для обучения (заголовки видео)
labels = data["category"].astype(int).tolist()  # Метки категорий

# Разделение на тренировочную и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Токенизация: преобразуем текст в формат, который можно передать в модель
def tokenize_texts(texts, labels):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Переносим токены на устройство
    labels = torch.tensor(labels).to(device)  # Переносим метки на устройство
    return tokens, labels

# Токенизируем тренировочные и тестовые данные
train_tokens, train_labels = tokenize_texts(train_texts, train_labels)
test_tokens, test_labels = tokenize_texts(test_texts, test_labels)

# Обучение модели
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def train_model(model, train_tokens, train_labels, epochs=3):
    # Подготовка данных для обучения
    train_data = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # Создаем DataLoader для мини-батчей

    # Настройка оптимизатора и функции потерь
    optimizer = AdamW(model.parameters(), lr=5e-5)  # Оптимизатор AdamW с маленьким шагом
    loss_fn = CrossEntropyLoss()  # Функция потерь для многоклассовой классификации

    model.train()  # Переводим модель в режим обучения
    for epoch in range(epochs):
        total_loss = 0  # Суммарная потеря за эпоху
        for batch in train_loader:
            input_ids, attention_mask, labels = batch  # Разделяем батч на токены и метки
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # Прогоняем через модель
            loss = loss_fn(outputs.logits, labels)  # Вычисляем потерю
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Обновляем параметры модели
            total_loss += loss.item()  # Добавляем потерю текущего батча
        print(f"Эпоха {epoch + 1}, Потеря: {total_loss:.4f}")  # Выводим потерю за эпоху

# Обучаем модель
train_model(model, train_tokens, train_labels, epochs=3)

# Оценка на тестовой выборке
def evaluate_model(model, test_tokens, test_labels):
    # Подготовка данных для оценки
    test_data = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)  # Загружаем данные для теста

    model.eval()  # Переводим модель в режим оценки
    predictions = []  # Список предсказаний
    true_labels = []  # Список истинных меток

    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        for batch in test_loader:
            input_ids, attention_mask, labels = batch  # Разделяем батч на токены и метки
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # Прогоняем через модель
            logits = outputs.logits  # Получаем логиты (до применения softmax)
            batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()  # Получаем предсказания (максимальные логиты)
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
evaluate_model(model, test_tokens, test_labels)
