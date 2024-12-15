import os
import re
import torch
import pdfplumber
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


# команды по примеру:
#page: 1, line: 2, start_char: 15, end_char: 21, type: word
#page: 1, line: 4, start_char: 10, end_char: 30, type: multiple_words
#page: 1, line: 4, start_char: 10, end_char: 30, type: sentence




def load_model(model_path, device):
    """
    Загрузка предобученной модели и токенизатора.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_text_from_pdf(pdf_path):
    """
    Извлечение текста из PDF-файла с помощью pdfplumber.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return pages
    except Exception as e:
        print(f"Ошибка при извлечении текста из PDF: {e}")
        return []


def find_masked_spans(line):
    """
    Поиск всех маскированных фрагментов в строке.
    Возвращает список кортежей (start_char, end_char, type).
    """
    masked_spans = []
    for match in re.finditer(r'(\[MASK\]+)', line):
        start, end = match.span()
        mask_length = len(match.group(1).split(']['))  # Количество [MASK]
        mask_count = line[start:end].count('[MASK]')

        # Определение типа маски
        if mask_count == 1:
            mask_type = "word"
        elif mask_count <= 3:
            mask_type = "multiple_words"
        else:
            mask_type = "sentence"

        masked_spans.append((start, end, mask_type))
    return masked_spans


def predict_masked_span(model, tokenizer, device, text, mask_start, mask_end):
    """
    Предсказание маскированного фрагмента текста.
    """
    # Создание ввода для модели
    inputs = tokenizer(text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Получение предсказаний для всех масок
    mask_indices = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predictions = []
    for mask_idx in mask_indices:
        mask_logits = logits[0, mask_idx, :]
        mask_probs = torch.softmax(mask_logits, dim=0)
        top_token_id = torch.argmax(mask_probs).item()
        predicted_token = tokenizer.decode([top_token_id]).strip()
        predictions.append(predicted_token)

    # Объединение предсказаний
    predicted_text = ' '.join(predictions)
    return predicted_text


def replace_masked_spans(line, masked_spans, predictions):
    """
    Замена маскированных фрагментов на предсказанные значения.
    """
    new_line = ""
    last_idx = 0
    for (span, pred) in zip(masked_spans, predictions):
        start, end, _ = span
        new_line += line[last_idx:start] + pred
        last_idx = end
    new_line += line[last_idx:]
    return new_line


def save_restored_pdf(restored_text, output_pdf_path):
    """
    Сохранение восстановленного текста в PDF-файл с помощью ReportLab.
    """
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    text_object = c.beginText(margin, height - margin)
    text_object.setFont("Helvetica", 12)

    for line in restored_text.split('\n'):
        text_object.textLine(line)

    c.drawText(text_object)
    c.save()


def inference(input_pdf_path, output_pdf_path, model_path):
    """
    Основная функция инференса.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели и токенизатора
    tokenizer, model = load_model(model_path, device)

    # Извлечение текста из PDF
    pages = extract_text_from_pdf(input_pdf_path)
    if not pages:
        print("Нет текста для обработки.")
        return
    page_num = 1  # Поскольку у нас одна страница
    text = pages[0]

    # Разбиение текста на строки
    lines = text.split('\n')
    restored_lines = []

    for line_num, line in enumerate(lines, start=1):
        masked_spans = find_masked_spans(line)
        predictions = []

        for span in masked_spans:
            start_char, end_char, mask_type = span
            # Предсказание
            predicted_text = predict_masked_span(model, tokenizer, device, line, start_char, end_char)
            predictions.append(predicted_text)
            # Логирование
            print(
                f"page: {page_num}, line: {line_num}, start_char: {start_char}, end_char: {end_char}, type: {mask_type}")

        if predictions:
            # Замена масок на предсказания
            restored_line = replace_masked_spans(line, masked_spans, predictions)
            restored_lines.append(restored_line)
        else:
            restored_lines.append(line)

    restored_text = '\n'.join(restored_lines)

    # Сохранение восстановленного текста в PDF
    save_restored_pdf(restored_text, output_pdf_path)
    print(f"Восстановленный PDF сохранён как {output_pdf_path}")


# Пример использования функции
if __name__ == '__main__':
    input_pdf = "dead soul_randomly_removed.pdf"
    output_pdf = "dead soul_restored.pdf"
    model_directory = "best_mlm_model_fold_1.pt"  # Укажите путь к вашей сохранённой модели

    # Проверка наличия модели
    if not os.path.exists(model_directory):
        print(f"Файл модели {model_directory} не найден.")
    else:
        inference(input_pdf, output_pdf, model_directory)
