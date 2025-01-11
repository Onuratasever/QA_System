import re

def calculate_page_scores(file_path, keywords, in_line1, in_line2, isActive, isDoubleActive):
    # Anahtar kelimeleri küçük harfe dönüştürme
    if isDoubleActive:
        #kırmızı yaz
        print("\033[91m" + "isDoubleActive" + "\033[0m")
    elif isActive:
        #mavi yaz
        print("\033[94m" + "isActive" + "\033[0m")    
    else:
        #sarı yaz
        print("\033[93m" + "isNotActive" + "\033[0m")
    keywords = [keyword.lower() for keyword in keywords]
    
    must_include_keyword = "kısa vade"
    # Dosyayı okuma
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Sayfa ayrıştırma: "Sayfa X" ile başlar, bir sonraki "Sayfa X+1" ile biter
    pages = re.split(r"\bSayfa \d+\b", content)  # Sayfa başlıklarına göre böl
    pages = [page.strip() for page in pages if page.strip()]  # Boş sayfaları temizle
    
    # Sayfa bazında puanlama
    nonzero_score_pages = []
    for page_number, page in enumerate(pages, start=1):
        score = 0
        # if must_include_keyword not in page.lower():
        #     continue  # Eğer 'kısa vade' sayfada yoksa bu sayfayı atla
        lines = page.split('\n')  # Sayfayı satırlara böl
        for line in lines:
            line = line.lower()  # Satırı küçük harfe dönüştür
            if isDoubleActive and in_line1 in line and in_line2 in line:
                actual_result = []
                actual_result.append((page_number, page, score))
                return actual_result
            elif isActive and in_line1 in line:
                actual_result = []
                actual_result.append((page_number, page, score))
                return actual_result
            # Anahtar kelimelerden birden fazlası aynı satırda geçiyor mu?
            match_count = sum(1 for keyword in keywords if keyword in line)
            if match_count > 1:  # Birden fazla anahtar kelime varsa
                score += 1
        if score > 0:  # Puanı 0'dan farklıysa listeye ekle
            nonzero_score_pages.append((page_number, page, score))

    # number of pages that have a score greater than 0
    # print("Number of pages that have a score greater than 0")
    # print(len(nonzero_score_pages))  
    return nonzero_score_pages



