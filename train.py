import gc
import numpy as np
import re
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Проверка импорта
try:
    import transformers
    print("transformers version:", transformers.__version__)
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    exit(1)

# Загрузка фраз
try:
    with open("generated_phrases.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
except FileNotFoundError:
    print("Файл generated_phrases.txt не найден.")
    exit(1)

# Предобработка
phrases = []
for line in lines:
    line = line.strip().lower()
    line = re.sub(r'[^\w\s]', '', line)
    if line:
        phrases.append(line)

print("Фраз в датасете:", len(phrases))

# Модель и токенизатор
MODEL_NAME = "cointegrated/rubert-tiny2"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Токенизатор загружен.")
except Exception as e:
    print(f"Ошибка токенизатора: {e}")
    exit(1)

# Подготовка данных
def prepare_data(phrases, max_length):
    inputs = []
    labels = []
    for phrase in phrases:
        words = phrase.split()
        for i in range(1, len(words)):
            input_text = ' '.join(words[:i])
            label = words[i]
            inputs.append(str(input_text))
            labels.append(label)
    return Dataset.from_dict({"text": inputs, "label": labels})

try:
    max_length = 16
    dataset = prepare_data(phrases, max_length)
    encoded_dataset = dataset.map(
        lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=max_length),
        batched=True
    )
except Exception as e:
    print(f"Ошибка при кодировании: {e}")
    exit(1)

# Проверка структуры датасета
print("Пример данных:", encoded_dataset[0])
print("Структура датасета:", encoded_dataset.features)

# Словарь меток
try:
    word_to_index = {word: idx for idx, word in
                     enumerate(sorted(list(set(' '.join(phrases).split())) + ['<pad>', '<unk>']))}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    encoded_dataset = encoded_dataset.map(lambda x: {'label': word_to_index.get(x['label'], word_to_index['<unk>'])})
except Exception as e:
    print(f"Ошибка при создании словаря меток: {e}")
    exit(1)

# Проверка меток
for i in range(5):
    print(f"Пример {i}: text={encoded_dataset[i]['text']}, label={encoded_dataset[i]['label']}, type(label)={type(encoded_dataset[i]['label'])}")

# Удаление ненужного поля text
encoded_dataset = encoded_dataset.remove_columns(['text'])

# Проверка структуры датасета после удаления
print("Структура датасета после удаления text:", encoded_dataset.features)

# Сохранение словарей
with open("word_to_index.json", "w", encoding="utf-8") as f:
    json.dump(word_to_index, f, ensure_ascii=False)
with open("index_to_word.json", "w", encoding="utf-8") as f:
    json.dump(index_to_word, f, ensure_ascii=False)

# Загрузка модели
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(word_to_index))
    print("Модель загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit(1)

# Аргументы тренировки
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_strategy="steps",  # Включение логирования
    logging_steps=100,
    save_total_limit=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=True,  # Автоматическое удаление ненужных столбцов
)

# Тренировка
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
    processing_class=tokenizer,  # Используем processing_class вместо tokenizer
)

try:
    trainer.train()
except Exception as e:
    print(f"Ошибка при обучении: {e}")
    exit(1)

# Сохранение
try:
    model.save_pretrained("rubert_tiny_autocomplete_model")
    tokenizer.save_pretrained("rubert_tiny_autocomplete_model")
    encoded_dataset.save_to_disk("encoded_dataset")
    print("Модель, токенизатор и датасет сохранены.")
except Exception as e:
    print(f"Ошибка при сохранении: {e}")
    exit(1)

# Очистка памяти
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()