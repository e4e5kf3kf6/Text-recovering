import json
import re
import os
import nltk
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import math
import pdfplumber
import random
import time
import psutil  # Для измерения потребления памяти
from datetime import datetime
import seaborn as sns  # Для построения heatmap

# Настройка стилей Seaborn
sns.set(style="whitegrid")

# Файлы с метриками
METRICS_FILE = "metrics.txt"
FINAL_METRICS_FILE = "final_metrics_GRISHA.txt"

# Удаление существующих файлов с метриками
if os.path.exists(METRICS_FILE):
    os.remove(METRICS_FILE)
if os.path.exists(FINAL_METRICS_FILE):
    os.remove(FINAL_METRICS_FILE)


def clean_text(text):
    """Очистка текста от URL и спецсимволов."""
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s\[\]]', ' ', text)
    return text.lower().strip()


def load_data(pdf_path, json_path):
    """Загрузка текста из PDF и информации о удалённых фрагментах из JSON."""
    try:
        # Извлечение текста из PDF
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]

        # Загрузка информации о удалениях из JSON
        with open(json_path, 'r', encoding='utf-8') as file:
            removals = json.load(file)

        # Применение масок к тексту на основе удалений
        for removal in removals:
            page_num = removal['page'] - 1  # Индексация с 0
            line_num = removal['line'] - 1
            start_char = removal['start_char']
            end_char = removal['end_char']
            removed_text = removal['removed_text']

            if page_num < len(pages):
                lines = pages[page_num].split('\n')
                if line_num < len(lines):
                    line = lines[line_num]
                    # Создание строки с маской
                    masked_line = line[:start_char] + '[MASK]' * (end_char - start_char) + line[end_char:]
                    lines[line_num] = masked_line
                    pages[page_num] = '\n'.join(lines)

        # Объединение всех страниц в один текст
        full_text = '\n'.join(pages)
        texts = full_text.split('\n')
        return texts
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None


class MaskedLMDataset(Dataset):
    """Набор данных для маскированного языкового моделирования."""

    def __init__(self, texts, tokenizer, max_length=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.inputs, self.attention_masks, self.labels = self.prepare_data()

    def prepare_data(self):
        inputs = []
        attention_masks = []
        labels = []
        for text in self.texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]

            # Создание маски
            labels_ids = input_ids.clone()
            probability_matrix = torch.full(labels_ids.shape, self.mask_prob)

            # Генерация специальной маски токенов
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                labels_ids.tolist(),
                already_has_special_tokens=True
            )
            probability_matrix = probability_matrix.masked_fill(
                torch.tensor(special_tokens_mask, dtype=torch.bool),
                value=0.0
            )

            # Определение маскированных индексов
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels_ids[~masked_indices] = -100  # Не вычислять градиенты для не-маскированных токенов

            # Замена маскированных токенов на [MASK]
            input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            inputs.append(input_ids)
            attention_masks.append(attention_mask)
            labels.append(labels_ids)

        input_ids = torch.stack(inputs)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        return input_ids, attention_masks, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


class CustomMaskedLMModel(nn.Module):
    """Кастомная модель для маскированного языкового моделирования."""

    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


def evaluate_model(model, dataloader, device):
    """Оценка модели на валидационной выборке."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    trues = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item()

            mask_positions = (labels != -100)
            if mask_positions.any():
                masked_labels = labels[mask_positions]
                logits = outputs.logits[mask_positions]
                predicted_ids = torch.argmax(logits, dim=-1)

                preds.extend(predicted_ids.cpu().tolist())
                trues.extend(masked_labels.cpu().tolist())
                total_count += len(masked_labels)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    if total_count > 0:
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='weighted')
        cm = confusion_matrix(trues, preds)
    else:
        acc = 0.0
        f1 = 0.0
        cm = None

    return avg_loss, acc, f1, cm, trues, preds


def plot_loss(train_losses, val_losses, fold):
    """Построение графика потерь."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training History Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs("metrics_plots", exist_ok=True)
    plt.savefig(f'metrics_plots/training_history_fold_{fold}.png')
    plt.close()


def plot_confusion_matrix(cm, fold, epoch, tokenizer, trues, preds):
    """Построение матрицы путаницы."""
    if cm is None or cm.size == 0:
        return

    all_labels = list(set(trues + preds))
    max_labels = 20
    if len(all_labels) > max_labels:
        token_counts = {}
        for t in trues:
            token_counts[t] = token_counts.get(t, 0) + 1
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:max_labels]
        top_tokens_ids = [x[0] for x in top_tokens]

        unique_sorted_labels = sorted(set(trues + preds))
        cm_full = cm
        filtered_labels = [l for l in unique_sorted_labels if l in top_tokens_ids]

        if len(filtered_labels) < 2:
            return

        label_to_idx = {l: i for i, l in enumerate(unique_sorted_labels)}
        idxs = [label_to_idx[l] for l in filtered_labels]

        cm = cm_full[np.ix_(idxs, idxs)]
        labels_for_plot = filtered_labels
    else:
        labels_for_plot = sorted(all_labels)

    xticks = [tokenizer.convert_ids_to_tokens(l) for l in labels_for_plot]
    yticks = [tokenizer.convert_ids_to_tokens(l) for l in labels_for_plot]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticks, yticklabels=yticks)
    plt.title(f'Confusion Matrix Fold {fold} Epoch {epoch}')
    plt.xlabel('Predicted Token')
    plt.ylabel('True Token')
    plt.tight_layout()
    filename = f'confusion_matrix_fold_{fold}_epoch_{epoch}.png'
    plt.savefig(f'metrics_plots/{filename}')
    plt.close()
    print(f"Матрица путаницы сохранена как metrics_plots/{filename}")


def save_metrics_to_file(filepath, info):
    """Сохранение информации о метриках в файл."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(info + '\n')


def log_performance(epoch_start_time, epoch_end_time, process):
    """Логирование времени обучения и потребления памяти."""
    epoch_time = epoch_end_time - epoch_start_time
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # в МБ
    return epoch_time, memory_usage


def save_sample_predictions(model, tokenizer, device, texts, fold, epoch, num_samples=3):
    """Сохранение примеров заполненных пропусков."""
    model.eval()
    sample_indices = random.sample(range(len(texts)), min(num_samples, len(texts)))
    samples = [texts[i] for i in sample_indices]

    predictions = []
    with torch.no_grad():
        for i, text in enumerate(samples):
            encoding = tokenizer(text, return_tensors='pt').to(device)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            masked_indices = (input_ids == tokenizer.mask_token_id)
            predicted_ids = torch.argmax(logits, dim=-1)
            input_ids[masked_indices] = predicted_ids[masked_indices]
            filled_text = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.append(filled_text)
            print(f"Sample {i + 1} Fold {fold} Epoch {epoch}:")
            print(f"Original: {text}")
            print(f"Filled: {filled_text}\n")

    # Сохранение примеров в файл
    with open(f'metrics_plots/sample_predictions_fold_{fold}_epoch_{epoch}.txt', 'w', encoding='utf-8') as f:
        for i, (orig, filled) in enumerate(zip(samples, predictions), 1):
            f.write(f"Sample {i} Fold {fold} Epoch {epoch}:\n")
            f.write(f"Original: {orig}\n")
            f.write(f"Filled: {filled}\n\n")


def compute_specific_metrics(metrics, models):
    """Вычисление и сохранение специфичных показателей эффективности."""
    summary = []
    for model in models:
        df = metrics[model]
        # Среднее время обучения на эпоху
        avg_training_time = df["Training Time (s)"].mean()
        # Среднее потребление памяти
        avg_memory = df["Memory Usage (MB)"].mean()
        # Финальные значения метрик
        final_accuracy = df["Final Accuracy"].iloc[-1]
        final_f1 = df["Final F1-Score"].iloc[-1]
        # Сохранение в список
        summary.append({
            "Model": model,
            "Avg Training Time per Epoch (s)": round(avg_training_time, 2),
            "Avg Memory Usage (MB)": round(avg_memory, 2),
            "Final Accuracy": round(final_accuracy, 4),
            "Final F1-Score": round(final_f1, 4)
        })
    # Преобразование в DataFrame
    summary_df = pd.DataFrame(summary)
    os.makedirs("metrics_plots", exist_ok=True)
    # Сохранение в CSV
    summary_df.to_csv("metrics_plots/specific_metrics_summary.csv", index=False)
    print("Специфичные показатели эффективности сохранены в metrics_plots/specific_metrics_summary.csv")

    # Визуализация
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Avg Training Time per Epoch (s)", hue="Model", data=summary_df, palette="viridis",
                legend=False)
    plt.title("Среднее время обучения на эпоху")
    plt.ylabel("Время (сек)")
    plt.xlabel("Модель")
    plt.tight_layout()
    plt.savefig("metrics_plots/avg_training_time.png")
    plt.close()
    print("График среднего времени обучения сохранён в metrics_plots/avg_training_time.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Avg Memory Usage (MB)", hue="Model", data=summary_df, palette="magma", legend=False)
    plt.title("Среднее потребление памяти")
    plt.ylabel("Память (MB)")
    plt.xlabel("Модель")
    plt.tight_layout()
    plt.savefig("metrics_plots/avg_memory_usage.png")
    plt.close()
    print("График среднего потребления памяти сохранён в metrics_plots/avg_memory_usage.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Final Accuracy", hue="Model", data=summary_df, palette="coolwarm", legend=False)
    plt.title("Итоговая точность")
    plt.ylabel("Accuracy")
    plt.xlabel("Модель")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("metrics_plots/final_accuracy.png")
    plt.close()
    print("График итоговой точности сохранён в metrics_plots/final_accuracy.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Final F1-Score", hue="Model", data=summary_df, palette="flare", legend=False)
    plt.title("Итоговый F1-Score")
    plt.ylabel("F1-Score")
    plt.xlabel("Модель")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("metrics_plots/final_f1_score.png")
    plt.close()
    print("График итогового F1-Score сохранён в metrics_plots/final_f1_score.png")


def main():
    """Основная функция обучения модели."""
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process = psutil.Process(os.getpid())

    # Пути к данным
    pdf_path = "dead soul_randomly_removed.pdf"
    json_path = "dead soul.json"

    # Загрузка и подготовка данных
    texts = load_data(pdf_path, json_path)
    if texts is None or len(texts) == 0:
        print("No data found or empty dataset.")
        return

    texts = [clean_text(t) for t in texts]

    # Список моделей для обучения
    models_to_train = [
        "DeepPavlov/rubert-base-cased",  # ruBERT
        "distilbert-base-uncased",  # DistilBERT (английская версия)
        "bert-base-multilingual-cased",  # Multilingual BERT
        "bert-base-uncased"  # Оригинальный BERT (английская версия без учета регистра)
    ]

    # Настройка K-Fold
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_results = []
    metrics = {}

    for model_name in models_to_train:
        print(f"\nStarting training for model: {model_name}")

        # Инициализация токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for fold, (train_index, val_index) in enumerate(kf.split(texts)):
            print(f"\nFold {fold + 1}/4 for model {model_name}")

            train_texts = [texts[i] for i in train_index]
            val_texts = [texts[i] for i in val_index]

            # Создание набора данных
            train_dataset = MaskedLMDataset(train_texts, tokenizer, max_length=128, mask_prob=0.15)
            val_dataset = MaskedLMDataset(val_texts, tokenizer, max_length=128, mask_prob=0.15)

            # Создание загрузчиков данных
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

            # Инициализация модели
            model = CustomMaskedLMModel(model_name)
            model.to(device)

            # Оптимизатор и скейлер для смешанной точности
            optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-6)
            scaler = GradScaler()

            best_val_loss = float('inf')
            early_stopping_counter = 0
            patience = 3  # Количество эпох для раннего останова

            train_losses = []
            val_losses = []
            val_accuracies = []
            val_f1_scores = []
            val_perplexities = []

            for epoch in range(5):
                epoch_start_time = time.time()
                model.train()
                total_train_loss = 0.0

                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                    if loss is None:
                        loss = torch.tensor(0.0, device=device)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_train_loss += loss.item()

                avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
                train_losses.append(avg_train_loss)

                # Оценка на валидационной выборке
                val_loss, val_acc, val_f1, cm, trues, preds = evaluate_model(model, val_loader, device)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1_scores.append(val_f1)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')  # чтобы избежать overflow
                val_perplexities.append(val_ppl)

                epoch_end_time = time.time()
                epoch_time, memory_usage = log_performance(epoch_start_time, epoch_end_time, process)

                # Логирование метрик
                info = (f"Model: {model_name}, Fold {fold + 1}, Epoch {epoch + 1}: "
                        f"Train Loss={avg_train_loss:.4f}, "
                        f"Val Loss={val_loss:.4f}, "
                        f"Val Acc={val_acc:.4f}, "
                        f"Val F1={val_f1:.4f}, "
                        f"Val PPL={val_ppl:.4f}, "
                        f"Epoch Time={epoch_time:.2f}s, "
                        f"Memory Usage={memory_usage:.2f}MB")
                print(info)
                save_metrics_to_file(METRICS_FILE, info)

                # Построение графиков потерь
                plot_loss(train_losses, val_losses, f"{model_name}_fold_{fold + 1}")

                # Построение матрицы путаницы
                plot_confusion_matrix(cm, f"{model_name}_fold_{fold + 1}", epoch + 1, tokenizer, trues, preds)

                # Сохранение примеров заполненных масок
                save_sample_predictions(model, tokenizer, device, val_texts, f"{model_name}_fold_{fold + 1}", epoch + 1,
                                        num_samples=3)

                # Ранний останов
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    # Сохранение лучшей модели
                    model_save_path = f'best_mlm_model_{model_name.replace("/", "_")}_fold_{fold + 1}.pt'
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Новая лучшая модель сохранена для фолда {fold + 1} модели {model_name}.\n")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered")
                        break

            # Сохранение итоговых результатов фолда
            fold_results.append({
                'model': model_name,
                'fold': fold + 1,
                'best_val_loss': best_val_loss,
                'last_val_acc': val_accuracies[-1] if val_accuracies else 0.0,
                'last_val_f1': val_f1_scores[-1] if val_f1_scores else 0.0,
                'last_val_ppl': val_perplexities[-1] if val_perplexities else float('inf')
            })
            metrics[model_name] = {
                'Train Loss': train_losses,
                'Val Loss': val_losses,
                'Val Acc': val_accuracies,
                'Val F1-Score': val_f1_scores,
                'Val PPL': val_perplexities,
                'Training Time (s)': [],  # Добавьте реальные данные, если необходимо
                'Memory Usage (MB)': []  # Добавьте реальные данные, если необходимо
            }

    # Вычисление и сохранение специфичных метрик
    compute_specific_metrics(metrics, models_to_train)

    # Запись итоговых результатов во внешний файл
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write("Final Results:\n")
        for result in fold_results:
            f.write(f"Model: {result['model']}, Fold {result['fold']}: Best Val Loss = {result['best_val_loss']:.4f}, "
                    f"Val Acc (last epoch) = {result['last_val_acc']:.4f}, "
                    f"Val F1 (last epoch) = {result['last_val_f1']:.4f}, "
                    f"Val PPL (last epoch) = {result['last_val_ppl']:.4f}\n")

    # Вывод итоговых результатов
    print("\nFinal Results:")
    for result in fold_results:
        print(f"Model: {result['model']}, Fold {result['fold']}: Best Val Loss = {result['best_val_loss']:.4f}, "
              f"Val Acc (last epoch) = {result['last_val_acc']:.4f}, "
              f"Val F1 (last epoch) = {result['last_val_f1']:.4f}, "
              f"Val PPL (last epoch) = {result['last_val_ppl']:.4f}")
