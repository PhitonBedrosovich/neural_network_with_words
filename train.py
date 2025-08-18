import gc
import numpy as np
import re
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import traceback
from transliterate import translit  # pip install transliterate

# Проверка импорта
try:
    import transformers

    print("transformers version:", transformers.__version__)
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    traceback.print_exc()
    exit(1)

# Загрузка датасета
try:
    df = pd.read_csv("dialogues.tsv", sep="\t", encoding="utf-8")
except FileNotFoundError:
    print("Файл dialogues.tsv не найден.")
    exit(1)
except Exception as e:
    print(f"Ошибка загрузки TSV: {e}")
    traceback.print_exc()
    exit(1)

# Предобработка: извлечение чистых фраз
phrases = []
for dialogue in df['dialogue']:
    # Удаляем HTML-теги
    clean_dialogue = re.sub(r'<[^>]+>', '', dialogue)
    # Разделяем на строки, игнорируя пустые
    lines = [line.strip() for line in clean_dialogue.split('\n') if line.strip()]
    for line in lines:
        # Проверяем, начинается ли строка с "Пользователь X: "
        if line.startswith("Пользователь"):
            # Извлекаем текст после "Пользователь X: "
            match = re.search(r'Пользователь \d+: (.*?)(?:\s*пользователь\s*|$)', line, re.IGNORECASE)
            if match:
                phrase = match.group(1).strip().lower()
                # Удаляем знаки препинания, оставляем буквы, цифры и пробелы
                phrase = re.sub(r'[^\w\s]', '', phrase)
                # Удаляем слово "пользователь" и лишние пробелы
                phrase = re.sub(r'\bпользователь\b', '', phrase).strip()
                # Пропускаем пустые фразы
                if phrase:
                    phrases.append(phrase)

# Отладочный вывод
print("Фраз в датасете:", len(phrases))
print("Примеры фраз:", phrases[:10])

# Проверка кодировки и токенизации фраз
for i, phrase in enumerate(phrases[:10]):
    words = phrase.split()
    print(f"Фраза {i}: {phrase}, слова: {words}")

# Модель и токенизатор
MODEL_NAME = "cointegrated/rubert-tiny2"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Токенизатор загружен.")
except Exception as e:
    print(f"Ошибка токенизатора: {e}")
    traceback.print_exc()
    exit(1)


# Подготовка данных
def prepare_data(phrases, max_length):
    inputs = []
    labels = []
    for phrase in phrases:
        words = phrase.split()
        if len(words) < 2:
            continue
        for i in range(1, min(len(words), max_length - 1)):
            input_text = ' '.join(words[max(0, i - (max_length - 2)):i])
            label = words[i]
            inputs.append(str(input_text))
            labels.append(label)
    return Dataset.from_dict({"text": inputs, "label": labels})


try:
    max_length = 16
    dataset = prepare_data(phrases, max_length)
    print(f"Датасет подготовлен, размер: {len(dataset)}")
    if len(dataset) == 0:
        print("Датасет пустой. Проверьте фразы на наличие хотя бы 2 слов в каждой.")
        exit(1)
    encoded_dataset = dataset.map(
        lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=max_length),
        batched=True,
        batch_size=256
    )
except Exception as e:
    print(f"Ошибка при кодировании: {e}")
    traceback.print_exc()
    exit(1)

# Словарь меток
try:
    # Whitelist и blacklist
    whitelist = {'hello', 'hi', 'ok', 'qq', 'bonjour', 'salam', 'hey', 'yo', 'id', 'end'}
    blacklist = {'ghbdtn', 'lfdfq', 'plhfdcndeqnt', 'pyfrjvbnmcz', 'zdaрова', 'zdravstvuyte', 'helloy'}

    # Словарь исправлений (расширьте по датасету)
    corrections = {
        'ghbdtn': 'привет',
        'lfdfq': 'давай',
        'plhfdcndeqnt': 'здравствуйте',
        'pyfrjvbnmcz': 'познакомимся',
        'helloy': 'hello',
        'zdaрова': 'здорова',
        'zdravstvuyte': 'здравствуйте',
        'privet': 'привет',
        'priveet': 'привет',
        'priivet': 'привет',
        # Добавьте вариации "привет" из вашего json: 'priiveeet', 'priveeeet' и т.д. → 'привет'
    }

    all_words = set()
    skipped_words = []

    for phrase in phrases:
        words = phrase.split()
        for word in words:
            # Нормализация: исправляем известные опечатки
            word = corrections.get(word.lower(), word.lower())

            # Фильтр для hex-хэшей перед translit
            if re.match(r'^[0-9a-f]{32,}$', word):
                skipped_words.append((word, "хэш перед translit"))
                continue

            # Пытаемся транслитерировать, если нет кириллицы
            if not re.search(r'[а-яА-Я]', word):
                try:
                    word = translit(word, 'ru')
                except:
                    pass  # Если не удалось, оставляем как есть

            # Дополнительный фильтр для translit'ированных хэшей (32+ символов из цифр и кириллических a-f эквивалентов)
            if len(word) >= 32 and re.match(r'^[0-9а-ёА-Ё]{32,}$', word.lower()):
                skipped_words.append((word, "translited hash"))
                continue

            # Фильтры (усиленные)
            if word in blacklist:
                skipped_words.append((word, "blacklist"))
                continue
            if word.isdigit():
                skipped_words.append((word, "число"))
                continue
            if len(word) < 2:
                skipped_words.append((word, "слишком короткое"))
                continue
            if len(word) > 50:
                skipped_words.append((word, "слишком длинное"))
                continue
            if re.match(r'^[\W_]+$', word):
                skipped_words.append((word, "только символы"))
                continue
            if re.match(r'^[0-9a-f]{32,}$', word):
                skipped_words.append((word, "хэш"))
                continue
            if word.lower() == 'пользователь' or not word:
                skipped_words.append((word, "пользователь или пустое"))
                continue
            # Пропускать латинские слова, если не в whitelist
            if not re.search(r'[а-яА-Я]', word) and word not in whitelist:
                skipped_words.append((word, "латинское без whitelist"))
                continue

            all_words.add(word)

    # Создание словарей
    word_to_index = {word: idx for idx, word in enumerate(sorted(list(all_words)))}
    word_to_index['<pad>'] = len(word_to_index)
    word_to_index['<unk>'] = len(word_to_index)
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    print("Размер очищенного словаря:", len(all_words))
    print("Примеры слов:", sorted(list(all_words))[:20])
    if skipped_words:
        print("Пропущенные слова:", skipped_words[:20])
except Exception as e:
    print(f"Ошибка при создании словаря меток: {e}")
    traceback.print_exc()
    exit(1)

# Маппинг меток
try:
    def map_labels(example):
        label = example['label']
        return {'label': word_to_index.get(label, word_to_index['<unk>'])}


    encoded_dataset = encoded_dataset.map(map_labels, batched=False)
except Exception as e:
    print(f"Ошибка при маппинге меток: {e}")
    traceback.print_exc()
    exit(1)

# Проверка меток
try:
    for i in range(min(5, len(encoded_dataset))):
        label_idx = encoded_dataset[i]['label']
        label_word = index_to_word.get(label_idx, "<не найдено>")
        print(f"Пример {i}: text={encoded_dataset[i]['text']}, label_idx={label_idx}, word={label_word}")
except Exception as e:
    print(f"Ошибка при проверке меток: {e}")
    traceback.print_exc()
    exit(1)

# Удаление ненужного поля text
encoded_dataset = encoded_dataset.remove_columns(['text'])

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
    traceback.print_exc()
    exit(1)

# Аргументы тренировки
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=True,
)

# Тренировка
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
    processing_class=tokenizer,
)

try:
    trainer.train()
except Exception as e:
    print(f"Ошибка при обучении: {e}")
    traceback.print_exc()
    exit(1)

# Сохранение
try:
    model.save_pretrained("rubert_tiny_autocomplete_model")
    tokenizer.save_pretrained("rubert_tiny_autocomplete_model")
    encoded_dataset.save_to_disk("encoded_dataset")
    print("Модель, токенизатор и датасет сохранены.")
except Exception as e:
    print(f"Ошибка при сохранении: {e}")
    traceback.print_exc()
    exit(1)

# Очистка памяти
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()