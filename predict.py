import json
import torch
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk

# Загрузка модели, токенизатора и датасета
try:
    model = AutoModelForSequenceClassification.from_pretrained("rubert_tiny_autocomplete_model")
    tokenizer = AutoTokenizer.from_pretrained("rubert_tiny_autocomplete_model")
    encoded_dataset = load_from_disk("encoded_dataset")
except Exception as e:
    print(f"Ошибка загрузки модели, токенизатора или датасета: {e}")
    print("Убедитесь, что папка 'rubert_tiny_autocomplete_model' и датасет 'encoded_dataset' существуют.")
    exit(1)

# Загрузка словаря
try:
    with open("index_to_word.json", "r", encoding="utf-8") as f:
        index_to_word = json.load(f)
except Exception as e:
    print(f"Ошибка загрузки словаря: {e}")
    exit(1)

# Улучшенная функция автодополнения
def get_suggestions(input_text, num_suggestions=5, max_length=16):
    input_text = input_text.lower().strip()
    
    # Улучшенная предобработка
    input_text = re.sub(r'\s+', ' ', input_text)
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
    except Exception as e:
        print(f"Ошибка при генерации подсказок: {e}")
        return []

    # Получаем топ предсказания
    top_indices = torch.topk(probabilities, num_suggestions * 3, dim=-1).indices[0].tolist()
    suggestions = []
    used_phrases = set()

    for idx in top_indices:
        word = index_to_word.get(str(idx), '<unk>')
        
        # Фильтрация нежелательных слов
        if (word in ['<unk>', '<pad>', ''] or 
            len(word) <= 1 or 
            word in input_text.split()):
            continue
            
        phrase = f"{input_text} {word}".strip()
        
        if phrase not in used_phrases and len(suggestions) < num_suggestions:
            suggestions.append(phrase)
            used_phrases.add(phrase)

            # Ограниченная генерация следующих слов
            current_text = phrase
            words_added = 0
            max_additional_words = 2  # Ограничиваем количество дополнительных слов
            
            for _ in range(max_additional_words):
                try:
                    inputs = tokenizer(current_text, return_tensors="pt", padding=True, truncation=True,
                                      max_length=max_length)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Берем топ-3 предсказания
                        top_k_indices = torch.topk(outputs.logits, 3, dim=-1).indices[0].tolist()
                        
                        # Ищем подходящее слово
                        next_word = None
                        for next_idx in top_k_indices:
                            candidate = index_to_word.get(str(next_idx), '<unk>')
                            if (candidate not in ['<unk>', '<pad>', ''] and 
                                len(candidate) > 1 and 
                                candidate not in current_text.split() and
                                candidate not in input_text.split()):
                                next_word = candidate
                                break
                        
                        if next_word:
                            current_text += f" {next_word}"
                            suggestions[-1] = current_text
                            words_added += 1
                        else:
                            break
                            
                except Exception as e:
                    break

    return suggestions[:num_suggestions]

# Улучшенный тест с несколькими примерами
test_phrases = [
    "привет что",
    "давай пойдем", 
    "где можно",
    "погода в москве",
    "хочу посмотреть",
    "купить билеты",
    "заказать еду",
    "как дела"
]

print("=== Тестирование улучшенной модели автодополнения ===\n")

for input_text in test_phrases:
    suggestions = get_suggestions(input_text, num_suggestions=3)
    print(f"Ввод: '{input_text}'")
    print("Подсказки:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    print("-" * 40)

# Дополнительная статистика
print("\n=== Статистика модели ===")
print(f"Размер словаря: {len(index_to_word)} слов")
print(f"Примеры слов в словаре: {list(index_to_word.values())[:10]}")

# Очистка памяти
if torch.cuda.is_available():
    torch.cuda.empty_cache()