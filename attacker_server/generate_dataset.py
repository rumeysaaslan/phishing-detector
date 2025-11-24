import os
import pandas as pd

# Sunucunun çalıştığı temel adres (tek makine varsayımıyla)
# Eğer 2 makine kullanıyorsan, buraya saldırgan makinenin IP'sini yaz: 'http://192.168.100.10:8080'
BASE_URL = 'http://127.0.0.1:8080'

# Hedef klasörler
phishing_folder = 'phishing_sites'
legitimate_folder = 'legitimate_sites'
output_csv_file = '../dataset.csv' # CSV dosyasını ana klasöre kaydet

data = []

# Phishing URL'lerini işle
print(f"İşleniyor: {phishing_folder}")
try:
    for filename in os.listdir(phishing_folder):
        if filename.endswith(".html"):
            # Dosya yolundan tam URL oluştur
            url = f"{BASE_URL}/{phishing_folder}/{filename}"
            # Veri listesine URL ve etiketi ekle (1 = phishing)
            data.append({'url': url, 'label': 1})
            print(f"  Eklendi (Phishing): {url}")
except FileNotFoundError:
    print(f"HATA: '{phishing_folder}' klasörü bulunamadı.")

# Meşru URL'lerini işle
print(f"\nİşleniyor: {legitimate_folder}")
try:
    for filename in os.listdir(legitimate_folder):
        if filename.endswith(".html"):
            # Dosya yolundan tam URL oluştur
            url = f"{BASE_URL}/{legitimate_folder}/{filename}"
            # Veri listesine URL ve etiketi ekle (0 = legitimate)
            data.append({'url': url, 'label': 0})
            print(f"  Eklendi (Legitimate): {url}")
except FileNotFoundError:
    print(f"HATA: '{legitimate_folder}' klasörü bulunamadı.")

# Toplanan veriyi bir Pandas DataFrame'e dönüştür
if data:
    df = pd.DataFrame(data)

    # Veriyi karıştır (Önemli! Modelin sırayı öğrenmemesi için)
    df = df.sample(frac=1).reset_index(drop=True)

    # DataFrame'i CSV dosyasına kaydet
    try:
        df.to_csv(output_csv_file, index=False, encoding='utf-8')
        print(f"\nVeri seti başarıyla '{output_csv_file}' dosyasına kaydedildi.")
        print(f"Toplam {len(df)} URL işlendi.")
    except Exception as e:
        print(f"HATA: CSV dosyası kaydedilemedi. {e}")
else:
    print("\nHiçbir URL işlenemedi. Klasörlerin doğru olduğundan ve içinde .html dosyaları bulunduğundan emin olun.")