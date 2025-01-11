from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Hugging Face Sentence Transformer modeli
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def sliding_window_line_chunks(lines, window_size=3, overlap=2):
    """
    Sliding window yöntemini satır tabanlı uygular.
    
    Args:
        lines (list of str): Satırların listesi.
        window_size (int): Her bir pencere kaç satır içerecek.
        overlap (int): Pencereler arasındaki örtüşme miktarı.
    
    Returns:
        list of str: Metin parçaları.
    """
    chunks = []
    for i in range(0, len(lines), window_size - overlap):
        chunk = "\n".join(lines[i:i + window_size])
        chunks.append(chunk)
    return chunks

def read_file_and_split_lines(file_path):
    """
    Dosyayı okuyup satır bazında ayırır.
    
    Args:
        file_path (str): Dosya yolu.
    
    Returns:
        list of str: Satırların listesi.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # Satır bazında oku
    return [line.strip() for line in lines if line.strip()]  # Boş satırları temizle

def start(question, pages,  window_size, overlap, output_file="relevant_chunks.txt", ):
    """
    Ana fonksiyon. Verilen soru ve sayfa listesine göre benzerlik hesaplar ve en alakalı 10 parçayı döner.

    Args:
        question (str): Kullanıcı sorusu.
        pages (list of str): Sayfa metinleri.
        output_file (str): Çıktı dosyasının adı.

    Returns:
        list of str: İlk 10 en benzer parça.
    """

    #pages'in kopyasını al
    pages_temp = pages.copy()

    # Sayfaları satır bazında birleştir ve temizle
    lines = [line.strip() for page in pages for line in page.split("\n") if line.strip()]
    #lines sayısını string olarak yaz
    # print("lines sayısı")
    # print(len(lines))
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    #cleaned lines sayısını stirng olarak yaz
    # print("cleaned_lines sayısı")
    # print(len(cleaned_lines))
    # Satırları parçalara böl
    chunks = sliding_window_line_chunks(cleaned_lines, window_size, overlap)
    #chunks içindeki eleman sayısı
    # print("chunks sayısı")
    # print(len(chunks))
    # Soru ve metin parçaları için embedding hesapla
    question_embedding = model.encode(question)
    chunk_embeddings = [model.encode(chunk) for chunk in chunks]

    # print("question_embedding")
    # print(question_embedding)
    # print("chunk_embeddings")
    # print(chunk_embeddings)

    if len(question_embedding) == 0 or len(chunk_embeddings) == 0:
        raise ValueError("question_embedding veya chunk_embeddings boş!")


    # Benzerlik skorlarını hesapla
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

    # Parçaları benzerlik skorlarına göre sırala
    ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)

    quarter = len(ranked_chunks) // 1
    # İlk 10 en alakalı parçayı seç
    top_10_chunks = ranked_chunks[:10]
    top_quarter_chunks = ranked_chunks[:quarter]
    if len(ranked_chunks) >= 100:
        top_200_chunks = ranked_chunks[:100]
    else:
        top_200_chunks = ranked_chunks
    #ilk chunk yazdır
    # ANSI kodu ile turuncu renk
    # print("\033[38;5;214m" + str(top_10_chunks[0][0]) + "\033[0m")

    seen_pages = []  
    # return top_200_chunks
    for chunk, _ in top_200_chunks:
        # Her chunk'ın ilk satırını sayfalarda ara
        first_line = chunk.split("\n")[0]  # Chunk'ın ilk satırını al
        second_line = chunk.split("\n")[1] if len(chunk.split("\n")) > 1 else ""  # İkinci satırını al (varsa)

        for page in pages_temp:
            if first_line in page and second_line in page:
                seen_pages.append((page, chunk))  # Sayfa zaten eklenmişse ekleme
    
    return seen_pages