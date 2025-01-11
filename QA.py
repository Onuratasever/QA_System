import json
import os
from tkinter import scrolledtext
import joblib
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from torch.nn.functional import softmax
import context_generator
from transformers import BertTokenizer, BertModel
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from sklearn.ensemble import IsolationForest

questions = {
    1: {"question": "Kilit yönetici personele ödenen ücreti toplam olarak açıklamış mı?", "question_number": 1},
    2: {"question": "Kilit yönetici personele ödenen ücreti çalışanlara sağlanan kısa vadeli faydalar kategorisi için açıklamış mı?", "question_number": 2},
    3: {"question": "Kilit yönetici personele ödenen ücreti diğer uzun vadeli faydalar kategorisi için açıklamış mı?", "question_number": 3},
    4: {"question": "Firmanın ilişkili taraflardan aldığı teminat, ipotek ve kefaletler tutarı açıklanmış mı?", "question_number": 4},
    5: {"question": "Firma hangi yılda kurulmuştur?", "question_number": 5},
    6: {"question": "Şirketin merkez adresi nedir?", "question_number": 6},
    7: {"question": "Şirketin merkez adresinin bulunduğu şehir neresidir?", "question_number": 7},
    8: {"question": "Personel sayısı kaçtır?", "question_number": 8},
    9: {"question": "Sorumlu denetçisi kimdir?", "question_number": 9},
    10: {"question": "Firmanın geçerli ve raporlama para birimi yazıyor mu?", "question_number": 10}
}

question_types = [
    "Firmanın geçerli ve raporlama para birimi yazıyor mu?",
    "Firmanın ilişkili taraflardan aldığı teminat, ipotek ve kefaletler tutarı açıklanmış mı?",
    "Kilit yönetici personele ödenen ücreti diğer uzun vadeli faydalar kategorisi için açıklamış mı?",
    "Kilit yönetici personele ödenen ücreti toplam olarak açıklamış mı?",
    "Kilit yönetici personele ödenen ücreti çalışanlara sağlanan kısa vadeli faydalar kategorisi için açıklamış mı?"
]

file_name_global = "ASELS_FR_2024.txt"

json_path = "standard_questions_data.json"

# Model ve tokenizer'ı yükle
model_path = "./fine_tuned_model_bert_tek"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model_bert_tek")

rf_model = joblib.load("ML_models/random_forest_model.joblib")
xgb_model = joblib.load("ML_models/xgboost_model.joblib")

embedding_tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')
embedding_model = BertModel.from_pretrained('dbmdz/bert-base-turkish-uncased')

save_context_rf = []
save_context_xgb = []
save_context = []

question_number = 2

context_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rf_index = 0
xgb_index = 0

def askBertQA():

    global save_context

    save_context = []
    for context in context_list:
        test_context = context

        # Girdiyi tokenize et
        inputs = tokenizer(
            questions[question_number]["question"],
            test_context,
            max_length=512,
            truncation="only_second",  # Context uzun olduğunda kırpma sadece context'te yapılır
            padding="max_length",      # Tüm girişler aynı uzunluğa getirilir
            return_tensors="pt"
        )


        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Cevap başlangıç ve bitiş indekslerini al
        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits)

        # Start ve end logits'leri softmax ile olasılığa dönüştür
        start_probs = softmax(start_logits, dim=1)
        end_probs = softmax(end_logits, dim=1)

        # Güven skoru
        confidence_score = (start_probs[0, answer_start_index] * end_probs[0, answer_end_index]).item()
        # print(f"Güven Skoru: {confidence_score:.2f}")

        # Cevabı al
        answer = tokenizer.decode(inputs['input_ids'][0][answer_start_index:answer_end_index + 1])
        # print("Cevap:", answer)

        save_context.append((test_context, answer, confidence_score))

    save_context.sort(key=lambda x: x[2], reverse=True)

    for i in save_context[:20]:
        print("------\nContext: ", i[0], "\nAnswer: ", i[1], "\nConfidence Score: ", i[2], "\n")

def create_question_vector(question, question_types):
    question_vector = np.zeros(len(question_types))  # Tüm sütunlar için sıfır vektörü oluştur
    if question in question_types:
        index = question_types.index(question)  # Sorunun listede bulunduğu index
        question_vector[index] = 1  # O indexi 1 yap
    return question_vector

def get_bert_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = embedding_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embedding.flatten()  # return as 1D array

def ask_RF_model(test_combined_embedding):
    # Get prediction from the model
    prediction = rf_model.predict([test_combined_embedding])
    # confidence score for the prediction
    probability = rf_model.predict_proba([test_combined_embedding])[0]

    return probability, prediction

def ask_XGB_model(test_combined_embedding):
    # Get prediction from the model
    prediction = xgb_model.predict([test_combined_embedding])
    # confidence score for the prediction
    probability = xgb_model.predict_proba([test_combined_embedding])[0]

    return probability, prediction

def ask_ML():

    global save_context_rf, save_context_xgb

    save_context_rf = []
    save_context_xgb = []

    rf_score = 0.8
    xgb_score = 0.8

    for context in context_list:
        test_context = context

        question_vector = create_question_vector(questions[question_number]["question"], question_types)

        test_context_embedding = get_bert_embedding(test_context)
        # Combine embeddings
        test_combined_embedding = np.hstack((test_context_embedding, question_vector)).flatten()

        rf_probability, rf_prediction = ask_RF_model(test_combined_embedding)

        if rf_prediction[0] == 1: # yes
            rf_score = rf_probability[1]
            save_context_rf.append((test_context, rf_score))

        xgboost_probability, xgboost_prediction = ask_XGB_model(test_combined_embedding)

        if xgboost_prediction[0] == 1: # yes
            xgb_score = xgboost_probability[1]
            save_context_xgb.append((test_context, xgb_score))
    
    save_context_rf.sort(key=lambda x: x[1], reverse=True)
    save_context_xgb.sort(key=lambda x: x[1], reverse=True)

    # print("--------------- Random Forest Model ----------------------")
    # for i in save_context_rf:
    #     print("------------------------------------\nContext: ", i[0], "\nConfidence Score: ", i[1], "\n")

    # print("--------------- XGBoost Model ----------------------")
    # for i in save_context_xgb:
    #     print("------------------------------------\nContext: ", i[0], "\nConfidence Score: ", i[1], "\n")

def update_standard_questions_data(score, file_name):
    # JSON dosyasının var olup olmadığını kontrol et
    if not os.path.exists(json_path):
        # Dosya yoksa başlangıç yapısı oluştur ve kaydet
        with open(json_path, 'w') as file:
            json.dump([], file)
    
    # Mevcut JSON verisini yükle
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    print("file name: ", file_name)
    # .txt uzantısını çıkar
    company_code, report_year = file_name.split("_")[0], file_name.split("_")[2].split(".")[0]
    
    print("company_code: *", company_code, "*")
    print("report_year: *", report_year, "*")
    #data len
    print("data before len: ", len(data))
    # Şirketin ilgili raporunun olup olmadığını kontrol et
    record_found = False
    if question_number == 1:
        question31 = "Q1"
    elif question_number == 2:
        question31 = "Q2"
    elif question_number == 3:
        question31 = "Q3"
    elif question_number == 4:
        question31 = "Q4"
    elif question_number == 10:
        question31 = "Q10"
    
    print("question31: ", question31)
    for record in data:
        if record['company_code'] == company_code and record['report_year'] == report_year:
            # İlgili rapor bulundu, sorunun skorunu güncelle
            record['questions'][question31] = score
            print("TRUEEEEEEEEEEEEEEEEE")
            record_found = True
            break
    
    if not record_found:
        # İlgili rapor yoksa yeni bir kayıt oluştur
        new_record = {
            "company_code": company_code,
            "report_year": report_year,
            "questions": {
                "Q1": -1,
                "Q2": -1,
                "Q3": -1,
                "Q4": -1,
                "Q10": -1
            }
        }
        # Yeni kayda gelen skorları ekle
        new_record['questions'][question31] = score
        data.append(new_record)
    
    # Güncellenmiş veriyi JSON dosyasına yaz
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
    print("data after len: ", len(data))
    print("Standard Questions Data Updated")

def detect_anomaly():
    # 1. JSON dosyasını oku
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 2. Özellik matrisini oluştur
    features = []
    company_info = []

    for record in data:
        company_code = record['company_code']
        report_year = record['report_year']
        questions = record['questions']
        
        # Her şirketin soru skorlarını al ve özellik vektörüne ekle
        feature_vector = [questions[q] for q in sorted(questions.keys())]  # Soruları sıralı al
        features.append(feature_vector)
        company_info.append((company_code, report_year))

    # 3. Özellikleri numpy array'e çevir
    features = np.array(features)

    # 4. Isolation Forest modeliyle anomali tespiti
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(features)

    # 5. Sonuçları yazdır
    for idx, prediction in enumerate(predictions):
        company_code, report_year = company_info[idx]
        anomaly_status = "Anomaly" if prediction == -1 else "Normal"
        print(f"Company: {company_code}, Year: {report_year}, Status: {anomaly_status}")

def main():
    global context_list, question_number
    folder_path = "C:/Users/onur/Desktop/Bitirme/Grad_Project_FR_txt_final"

    # Klasördeki tüm .txt dosyalarını bul
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    max_file_count = 0
    for file_name in txt_files:
        if max_file_count >= 400:
            break
        #5 kere dönecek bir for loop
        for i in range(1, 6): # burada 6. index dahil değil
            if i == 1 or i == 2 or i == 3 or i == 4:
                question_number = i
            if i == 5:
                question_number = 10
            print("question_number: ", question_number)
            context_list = context_generator.start(file_name, question_number)

            print("len of context_list: ", len(context_list))

            if question_number == 1 or question_number == 2 or question_number == 3 or question_number == 4 or question_number == 10:
                ask_ML()
                if save_context_rf and save_context_xgb:
                    update_standard_questions_data((save_context_rf[0][1] + save_context_xgb[0][1]) / 2, file_name)
                    # return "Evet"
                else:
                    update_standard_questions_data(0, file_name)
                    # return "Hayır"
            else:
                askBertQA()
                if save_context:
                    return save_context[0][1]
                else:
                    return "Not mentioned"

# def select_file():
#     global file_name
#     file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
#     if file_path:
#         file_label.config(text=f"Selected file: {file_path}")
#         file_name = file_path
#     else:
#         file_label.config(text="No file selected")

# # Soru seçildiğinde her şeyi sıfırla
# def update_question_number(*args):
#     global context_index, question_number
#     context_index = 0  # Context indexi sıfırla

#     # Tüm frame'leri gizle
#     model_menu.pack_forget()
#     rf_xgb_frame.pack_forget()
#     rf_nav_frame.pack_forget()
#     bert_frame.pack_forget()
#     bert_nav_frame.pack_forget()
#     context_label.pack_forget()
#     get_answer_label.pack_forget()
#     answer_label.config(text="")

#     selected_question = question_var.get()
    
#     for q_id, q_data in questions.items():
#         if q_data["question"] == selected_question:
#             selected_question_number.set(q_data["question_number"])
#             question_number = q_data['question_number']
#             print(f"Selected Question Number: {q_data['question_number']}")
#             question_id_label.config(text=f"Selected Question Number: {q_data['question_number']}")
            
#             # ML veya BERT seçimine göre butonları göster/gizle
#             if q_data["question_number"] in [1, 2, 3, 4, 10]:
#                 model_menu.pack()  # Model seçimi menüsünü göster
#                 rf_xgb_frame.pack()  # Show Context butonunu göster
#                 get_answer_label.pack()  # Get Answer butonunu göster
#             else:
#                 bert_frame.pack()  # BERT için Show Context butonunu göster
#                 get_answer_label.pack()  # Get Answer butonunu göster
#             break

# # Context gösterme
# def show_context(context_list, model_name):
#     global context_index
#     if not context_list:
#         context_label.config(text=f"No context found for the selected question.")
#         return
#     if 0 <= context_index < len(context_list):
#         if model_name == "ml":
#             context, score = context_list[context_index]
#             context_label.config(text=f"Context: {context}\n (Score: {score:.5f})")
#         else:
#             context, answer, score = context_list[context_index]
#             context_label.config(text=f"Context: {context}\nAnswer: {answer} (Confidence Score: {score:.5f})")
#         context_label.pack()

# # BERT için Show Context butonuna tıklanınca Previous/Next butonlarını göster
# def show_bert_context():
#     show_context(save_context, "bert")
#     bert_nav_frame.pack()  # Previous ve Next butonlarını göster

# # ML için Show Context butonuna tıklanınca Previous/Next butonlarını göster
# def show_ml_context():
#     selected_model = model_var.get()
#     if selected_model == "Random Forest":
#         show_context(save_context_rf, "ml")
#     elif selected_model == "XGBoost":
#         show_context(save_context_xgb, "ml")
#     else:
#         messagebox.showwarning("Warning", "Please select a valid model.")
#     rf_nav_frame.pack()  # ML için Previous ve Next butonlarını göster

# # İleri/geri butonları
# def next_context(context_list, model):
#     global context_index
#     if context_index < len(context_list) - 1:
#         context_index += 1
#         show_context(context_list, model)

# def prev_context(context_list, model):
#     global context_index
#     if context_index > 0:
#         context_index -= 1
#         show_context(context_list, model)

# def select_model(selected_model):
#     global context_index
#     context_index = 0  # zero
#     context_label.config(text="")

# def get_answer():
#     answer = main()
#     answer_label.config(text=answer)

# # Tkinter UI
# root = tk.Tk()
# root.title("Question Answering System")

# # Seçilen soru ID'sini göstermek için değişken
# selected_question_number = tk.IntVar(value=0)

# tk.Button(root, text="Select File", command=select_file).pack()
# file_label = tk.Label(root, text="No file selected")
# file_label.pack()

# # Soru seçimi için dropdown menü
# question_var = tk.StringVar(value="Select Question")
# question_menu = tk.OptionMenu(root, question_var, *[q["question"] for q in questions.values()], command=update_question_number)
# question_menu.pack()


# # Seçilen soru numarasını gösteren label
# question_id_label = tk.Label(root, text="Selected Question Number: 0")
# question_id_label.pack_forget()

# # Model seçimi için dropdown menü (başlangıçta gizli)
# model_var = tk.StringVar(value="Select Model")
# model_menu = tk.OptionMenu(root, model_var, "Random Forest", "XGBoost", command=select_model)
# model_menu.pack_forget()

# # RF/XGB için Show Context butonu
# rf_xgb_frame = tk.Frame(root)
# tk.Button(rf_xgb_frame, text="Show ML Context", command=show_ml_context).pack()

# # RF/XGB için Previous/Next butonları (başlangıçta gizli)
# rf_nav_frame = tk.Frame(root)
# tk.Button(rf_nav_frame, text="Previous ML Context", command=lambda: prev_context(save_context_rf if model_var.get() == "Random Forest" else save_context_xgb, "ml")).pack()
# tk.Button(rf_nav_frame, text="Next ML Context", command=lambda: next_context(save_context_rf if model_var.get() == "Random Forest" else save_context_xgb, "ml")).pack()

# # BERT için Show Context butonu
# bert_frame = tk.Frame(root)
# tk.Button(bert_frame, text="Show BERT Context", command=show_bert_context).pack()

# # BERT için Previous/Next butonları (başlangıçta gizli)
# bert_nav_frame = tk.Frame(root)
# tk.Button(bert_nav_frame, text="Previous BERT Context", command=lambda: prev_context(save_context, "bert")).pack()
# tk.Button(bert_nav_frame, text="Next BERT Context", command=lambda: next_context(save_context, "bert")).pack()

# # Context gösterimi (başlangıçta gizli)
# context_label = tk.Label(root, text="Context will appear here")

# # Create and place the answer text area
# answer_label = tk.Label(root, text="Answer:", width=100, height=2, bg="white", anchor="w", justify="left")
# answer_label.pack(pady=10)

# # Get Answer
# tk.Button(root, text="Get Answer", command=get_answer).pack()
# get_answer_label = tk.Label(root)
# get_answer_label.pack_forget()

# root.mainloop()

main()