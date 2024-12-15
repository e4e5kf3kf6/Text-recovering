import pdfplumber
import json
import random
import re
from reportlab.pdfgen import canvas
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4


PROB_REMOVE_WORD = 0.03  # Вероятность удаления одного слова
PROB_REMOVE_SENTENCE = 0.005  # Вероятность удаления одного предложения
PROB_REMOVE_MULTIPLE_WORDS = 0.01  # Вероятность удаления 2-3 слов подряд

# замени названия для второй книги

INPUT_PDF = "dead soul.pdf"
JSON_OUTPUT = "dead soul.json"
OUTPUT_PDF = "dead soul_randomly_removed.pdf"



def split_into_sentences(line):
    sentences = re.split(r'([.?!])', line)
    if len(sentences) < 2:
        return [line]
    merged = []
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i].strip()
        p = sentences[i + 1].strip()
        merged.append((s + p).strip())
    return [s for s in merged if s]


def remove_random_fragment(line, page_num, line_num, removals):
    r = random.random()
    if r < PROB_REMOVE_SENTENCE:
        sentences = split_into_sentences(line)
        if len(sentences) > 0:
            sent_id = random.randint(0, len(sentences) - 1)
            sentence = sentences[sent_id]

            start_idx = line.find(sentence)
            end_idx = start_idx + len(sentence)
            if start_idx != -1:
                removed_text = line[start_idx:end_idx]
                space_str = " " * len(removed_text)
                new_line = line[:start_idx] + space_str + line[end_idx:]
                removals.append({
                    "page": page_num,
                    "line": line_num,
                    "start_char": start_idx,
                    "end_char": end_idx,
                    "removed_text": removed_text,
                    "type": "sentence"
                })
                return new_line

    r = random.random()
    if r < PROB_REMOVE_MULTIPLE_WORDS:
        words = line.split()
        if len(words) > 3:
            count = random.randint(2, 3)  # удалим 2 или 3 слова
            start_word_id = random.randint(0, len(words) - count)
            removed_words = words[start_word_id:start_word_id + count]
            removed_str = " ".join(removed_words)
            pattern = r'\b' + re.escape(removed_words[0]) + r'\b'
            match_iter = list(re.finditer(pattern, line))
            if match_iter:
                sub_text = " ".join(removed_words)
                start_idx = line.find(sub_text)
                if start_idx != -1:
                    end_idx = start_idx + len(sub_text)
                    space_str = " " * (end_idx - start_idx)
                    new_line = line[:start_idx] + space_str + line[end_idx:]
                    removals.append({
                        "page": page_num,
                        "line": line_num,
                        "start_char": start_idx,
                        "end_char": end_idx,
                        "removed_text": removed_str,
                        "type": "multiple_words"
                    })
                    return new_line

    r = random.random()
    if r < PROB_REMOVE_WORD:
        words = line.split()
        if len(words) > 0:
            word_id = random.randint(0, len(words) - 1)
            removed_word = words[word_id]

            pattern = r'\b' + re.escape(removed_word) + r'\b'
            match_iter = list(re.finditer(pattern, line))
            if match_iter:
                m = match_iter[0]
                start_idx = m.start()
                end_idx = m.end()
                removed_text = line[start_idx:end_idx]
                space_str = " " * (end_idx - start_idx)
                new_line = line[:start_idx] + space_str + line[end_idx:]
                removals.append({
                    "page": page_num,
                    "line": line_num,
                    "start_char": start_idx,
                    "end_char": end_idx,
                    "removed_text": removed_text,
                    "type": "word"
                })
                return new_line

    return line



def create_pdf_from_text(text_pages, output_pdf):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_filename = 'accid___(Ru)_0.ttf'
    font_path = os.path.join(script_dir, font_filename)

    if not os.path.exists(font_path):
        print(
            f"Ошибка: Файл шрифта '{font_filename}' не найден в директории '{script_dir}'. Пожалуйста, убедитесь, что шрифт находится в правильной директории.")
        return
    try:
        pdfmetrics.registerFont(TTFont('AccidentalPresidency', font_path))
        print(f"Шрифт 'AccidentalPresidency' успешно зарегистрирован.")
    except Exception as e:
        print(f"Ошибка при регистрации шрифта: {e}")
        return

    c = canvas.Canvas(output_pdf, pagesize=A4)
    c.setFont("AccidentalPresidency", 12)
    y = 800

    for page_num, page in enumerate(text_pages, start=1):
        for line_num, line in enumerate(page, start=1):
            if not isinstance(line, str):
                print(f"Предупреждение: Строка {line_num} на странице {page_num} не является строкой. Пропуск.")
                continue
            try:
                c.drawString(50, y, line)
            except Exception as e:
                print(f"Ошибка при выводе строки на странице {page_num}, строка {line_num}: {e}")
                c.drawString(50, y, "[Ошибка отображения строки]")
            y -= 15
            if y < 50:
                c.showPage()
                c.setFont("AccidentalPresidency", 12)
                y = 800
        c.showPage()
        c.setFont("AccidentalPresidency", 12)
        y = 800
    c.save()




def main():

    if not os.path.exists(INPUT_PDF):
        print(f"Ошибка: Файл {INPUT_PDF} не найден.")
        return

    removals = []
    with pdfplumber.open(INPUT_PDF) as pdf:
        text_pages = []
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text is None:
                text_pages.append([])
                continue
            lines = text.split('\n')
            new_page_lines = []
            for line_num, line in enumerate(lines, start=1):
                new_line = remove_random_fragment(line, page_num, line_num, removals)
                new_page_lines.append(new_line)
            text_pages.append(new_page_lines)
    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(removals, f, ensure_ascii=False, indent=4)

    create_pdf_from_text(text_pages, OUTPUT_PDF)

    print("Набор данных создан:")
    print(f"- Модифицированный PDF: {OUTPUT_PDF}")
    print(f"- JSON файл с информацией о удалённых текстах: {JSON_OUTPUT}")
    print("Теперь у вас есть набор данных, состоящий из:")
    print("1) Оригинальный PDF (dead soul.pdf)")
    print("2) Модифицированный PDF с пропусками (dead soul_randomly_removed.pdf)")
    print("3) JSON файл с деталями удалённых текстов (dead soul.json)")


if __name__ == "__main__":
    main()
