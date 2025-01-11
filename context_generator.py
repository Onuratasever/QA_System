import os
import sliding_window
import short_pages
import pandas as pd
import json

MAX_FILES = 400
LOG_FILE_PATH = ""

keywords = []
question = ""
in_line1 = ""
in_line2 = ""
active = False
isDoubleActive = False
question_number = 0

q1_keywords = ["yönetici", "üst", "düzey", "kadro", "genel müdür", "ortak", "fayda", "huzur", "menfaat", "yönetim", "ücretler", "öde"]
q2_keywords = ["kısa vade", "sağlanan", "fayda", "ilişkin","karşılık", "menfaat", "çalışan", "kısa", "vade"]
q3_keywords = ["uzun vade", "sağlanan", "fayda", "ilişkin", "karşılık", "menfaat", "çalışan", "uzun", "vade"]
q4_keywords = ["rehin", "teminat", "ipotek", "kefalet", "depozito", "veril"]
q5_keywords = ["yılında", "kurulmuş", "tescil edil", "kuruluş", "tarih"]
q6_keywords = ["adres", "merkez adres", "merkez", "kayıtlı adres", "şirket merkezi", "yönetim merkezi", "blok", "kat", "cad.", "cadde"]
q7_keywords = ["adres", "merkez adres", "merkez", "kayıtlı adres", "şirket merkezi", "yönetim merkezi", "blok", "kat", "cad.", "cadde"]
q8_keywords = ["personel", "çalışan", "sayısı", "çalışan sayısı", "çalışanlar", "çalışanlar sayısı"]
q9_keywords = ["sorumlu", "denetçi", "başdenetçi"]
q10_keywords = ["fonksiyonel", "işlevsel", "raporlama", "sunum", "para birimi", "Türk lira"]

excel_file_path = ""  # Buraya Excel dosyanızın yolunu yazın

csv_headers = ["context", "question", "label"]

output_file = ""

total_data = 0
questions = {
        1: {"log_file": "log_generating_yes_q1.txt", "keywords": "keywords_q1", "question": "kilit yönetici personele ödenen ücreti toplam olarak  açıklamış mı?", "excel_file": "excels/q1.xlsx", "output_file": "outputs/q1_neg_labeled.json", "in_line1": "", "in_line2": ""},
    2: {"log_file": "log_generating_yes_q2.txt", "keywords": "keywords_q2", "question": "kilit yönetici personele ödenen ücreti çalışanlara sağlanan kısa vadeli faydalar kategorisi için açıklamış mı?", "excel_file": "excels/q2.xlsx", "output_file": "outputs/q2_positive_labeled.json", "in_line1": "kısa vadeli karşılıklar", "in_line2": ""},
    3: {"log_file": "log_generating_yes_q3.txt", "keywords": "keywords_q3", "question": "kilit yönetici personele ödenen ücreti diğer uzun vadeli faydalar  kategorisi için açıklamış mı?", "excel_file": "excels/q3.xlsx", "output_file": "outputs/q3_positive_labeled.json", "in_line1": "uzun vadeli karşılıklar", "in_line2": ""},
    4: {"log_file": "log_generating_yes_q4.txt", "keywords": "keywords_q4", "question": "Firmanın ilişkili taraflardan aldığı teminat, ipotek ve kefaletler tutarı nedir?", "excel_file": "excels/q4.xlsx", "output_file": "outputs/q4_positive_labeled.json", "in_line1": "teminat", "in_line2": "ipotek"},
    5: {"log_file": "log_generating_yes_q5.txt", "keywords": "keywords_q5", "question": "Firma hangi yılda kurulmuştur ?", "excel_file": "excels/q5.xlsx", "output_file": "outputs/q5_positive_labeled.json", "in_line1": "yılında", "in_line2": "kurulmuştur"},
    6: {"log_file": "log_generating_yes_q6.txt", "keywords": "keywords_q6", "question": "Şirketin merkez adresi nedir?", "excel_file": "excels/q6.xlsx", "output_file": "outputs/q6_positive_labeled.json", "in_line1": "merkez adres", "in_line2": ""},
    7: {"log_file": "log_generating_yes_q7.txt", "keywords": "keywords_q7", "question": "Şirketin merkez adresinin bulunduğu şehir nedir?", "excel_file": "excels/q7.xlsx", "output_file": "outputs/q7_positive_labeled.json", "in_line1": "merkez adres", "in_line2": ""},
    8: {"log_file": "log_generating_yes_q8.txt", "keywords": "keywords_q8", "question": "Personel sayısı kaçtır?", "excel_file": "excels/q8.xlsx", "output_file": "outputs/q8_positive_labeled.json", "in_line1": "çalışan sayısı", "in_line2": ""},
    9: {"log_file": "log_generating_yes_q9.txt", "keywords": "keywords_q9", "question": "Sorumlu denetçisi kimdir?", "excel_file": "excels/q9.xlsx", "output_file": "outputs/q9_positive_labeled.json", "in_line1": "sorumlu denetçi", "in_line2": ""},
    10: {"log_file": "log_generating_yes_q10.txt", "keywords": "keywords_q10", "question": "Firmanın geçerli ve raporlama para birimi yazıyor mu?", "excel_file": "excels/q10.xlsx", "output_file": "outputs/q10_positive_labeled.json", "in_line1": "para birimi", "in_line2": "türk"}
}

def setup_globals(number):
    global keywords, question, in_line1, in_line2, active,excel_file_path, isDoubleActive, output_file, LOG_FILE_PATH, question_number
    
    # print("Soru numarasını girin: ")
    question_number = number
    LOG_FILE_PATH = questions[question_number]["log_file"]
    keywords = globals().get(f"q{question_number}_keywords")
    question = questions[question_number]["question"]
    excel_file_path = questions[question_number]["excel_file"]
    output_file = questions[question_number]["output_file"]
    in_line1 = questions[question_number]["in_line1"]
    in_line2 = questions[question_number]["in_line2"]

    # print(question_number)
    # print(LOG_FILE_PATH)
    # print(keywords)
    # print(question)
    # print(excel_file_path)
    # print(output_file)
    # print(in_line1)
    # print(in_line2)

    # cevap = input("devem etmek istiyor musunuz? (e/h): ")
    # if cevap == "h":
    #     exit()
    if in_line1 != "" and in_line2 != "":
        isDoubleActive = True
    else:
        isDoubleActive = False
    if in_line1 != "" and in_line2 == "":
        active = True
    if in_line1 == "":
        active = False

def mark_file_as_processed(file_name):
    """
    İşlenmiş dosyayı log dosyasına ekler.

    Args:
        file_name (str): İşlenmiş dosyanın adı.
    """
    print(f"{file_name} işlendi. Log dosyasına ekleniyor...")
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"{file_name}\n")

def initialize_log_file():
    """
    Log dosyasını oluşturur (eğer yoksa).
    """
    if not os.path.exists(LOG_FILE_PATH):
        open(LOG_FILE_PATH, "w").close()

def is_file_processed(file_name):
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as log_file:
        processed_files = log_file.readlines()
        return file_name in (line.strip() for line in processed_files)

def get_relevant_txt(file_path, answer):
    pages = []
    related_chunks = []

    nonzero_score_pages = short_pages.calculate_page_scores(file_path, keywords, in_line1, in_line2, active, isDoubleActive)  # Sayfa puanlarını hesapla

    for page_number, page_content, score in nonzero_score_pages:
        # Sayfa başlığı ve içeriğini oluştur
        page_text = f"Sayfa {page_number}:\n{page_content}\n==========\n"
        pages.append(page_text)

    if pages:
        chunks =sliding_window.start(question, pages, 5, 4)
        
        print("chunks len")
        print(len(chunks))
        # return [chunk[1] for chunk in chunks]
        answer_parts = answer.split()
        score = 0
        max_score = 0
        max_index = -1
        for idx, (page, chunk) in enumerate(chunks):
            for keyword in keywords:
                if keyword in chunk:  # Eğer answer parçası page içinde bulunursa
                    score += 1
            related_chunks.append((chunk, score))
            score = 0
    related_chunks = sorted(related_chunks, key=lambda x: x[1], reverse=True) # Sort by score from high to low
    # return highest 30 chunks not their score if there are more than 30 chunks
    if len(related_chunks) > 30:
        return [chunk[0] for chunk in related_chunks[:30]]
    else:
        return [chunk[0] for chunk in related_chunks]

def generate_vectors(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    # file_path = file_name
    print("Dosya yolu: ", file_path)
    related_chunks = get_relevant_txt(file_path, question)
    # write related chunks to do file by seperating with =======
    with open("related_chunks.txt", "w", encoding="utf-8") as file:
        for chunk in related_chunks:
            file.write(chunk)
            file.write("\n==========\n")
    return related_chunks

def start(file_name, number):
    setup_globals(number)
    # initialize_log_file()

    # print("Dosya adı: ", file_name)
    folder_path = "C:/Users/onur/Desktop/Bitirme/Grad_Project_FR_txt_final"

    return generate_vectors(folder_path, file_name)
