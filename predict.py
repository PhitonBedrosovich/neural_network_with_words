import json
import torch
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

# Функция автодополнения
def get_suggestions(input_text, num_suggestions=3, max_length=16):  # Увеличен max_length для совместимости с train.py
    input_text = input_text.lower().strip()
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
    except Exception as e:
        print(f"Ошибка при генерации подсказок: {e}")
        return []

    top_indices = torch.topk(probabilities, num_suggestions, dim=-1).indices[0].tolist()
    suggestions = []
    used_phrases = set()

    for idx in top_indices:
        word = index_to_word.get(str(idx), '<unk>')
        phrase = f"{input_text} {word}".strip()
        if phrase not in used_phrases:
            suggestions.append(phrase)
            used_phrases.add(phrase)

            # Генерация следующих слов
            current_text = phrase
            for _ in range(max_length - 1):
                try:
                    inputs = tokenizer(current_text, return_tensors="pt", padding=True, truncation=True,
                                      max_length=max_length)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        next_idx = torch.argmax(outputs.logits, dim=-1).item()
                    next_word = index_to_word.get(str(next_idx), '<unk>')
                    if next_word not in current_text.split():
                        current_text += f" {next_word}"
                        suggestions[-1] = current_text
                    else:
                        break
                except Exception as e:
                    print(f"Ошибка при генерации следующего слова: {e}")
                    break

    return suggestions[:num_suggestions]

# Тест
input_text = "привет что"
suggestions = get_suggestions(input_text)
print(f"\nВвод: {input_text}")
print("Подсказки:")
for i, suggestion in enumerate(suggestions, 1):
    print(f"{i}. {suggestion}")

# Очистка памяти
if torch.cuda.is_available():
    torch.cuda.empty_cache()