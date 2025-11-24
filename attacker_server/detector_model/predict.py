import os
import numpy as np
import pandas as pd
import joblib  # Scaler'ı yüklemek için
import tensorflow as tf
import warnings # Uyarıları gizlemek için

# Özellik çıkarma fonksiyonumuzu içeren dosyayı import et
try:
    # predict.py ve features.py aynı 'detector_model' klasöründe olduğu için doğrudan import edebiliriz
    from features import extract_features
except ImportError:
    print("HATA: 'features.py' dosyası aynı klasörde bulunamadı.")
    print("      'predict.py' dosyasının 'detector_model' klasöründe olduğundan emin olun.")
    exit()

# Gereksiz TensorFlow ve diğer uyarıları gizleyelim (isteğe bağlı)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sadece hataları göster
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# --- Kaydedilmiş Dosyaların Yolları ---
# Bu script 'detector_model' içinde çalıştırılacağı için, dosya isimleri yeterli
SCALER_PATH = os.path.join('detector_model', 'scaler.joblib')
MODEL_PATH = os.path.join('detector_model', 'phishing_detector_model.keras')
# ---

# --- Model ve Scaler Yükleme ---
print(">>> Model ve Scaler yükleniyor...")
MODEL = None
SCALER = None
FEATURE_NAMES = None # Modelin eğitildiği özellik isimleri

# Scaler'ı yükle
if not os.path.exists(SCALER_PATH):
    print(f"HATA: Scaler dosyası bulunamadı: {SCALER_PATH}")
    print("      Lütfen önce 'train_detector.py' scriptini çalıştırarak modeli eğitin.")
    exit()
try:
    SCALER = joblib.load(SCALER_PATH)
    print(f"   + Scaler '{SCALER_PATH}' başarıyla yüklendi.")
    # Scaler'dan özellik isimlerini almayı dene (eğer varsa)
    if hasattr(SCALER, 'feature_names_in_'):
        FEATURE_NAMES = list(SCALER.feature_names_in_)
        print(f"   + Scaler {len(FEATURE_NAMES)} özellik bekliyor.")
    elif hasattr(SCALER, 'n_features_in_'):
         num_feat = SCALER.n_features_in_
         print(f"   ! Uyarı: Scaler'dan özellik isimleri alınamadı, sadece sayı ({num_feat}) biliniyor.")
         print("     Özellik sırasının eğitimdekiyle aynı olması kritik!")
    else:
         print("   ! Uyarı: Scaler'dan özellik sayısı/isimleri alınamadı.")

except Exception as e:
    print(f"HATA: Scaler yüklenirken hata: {e}")
    exit()

# Modeli yükle
if not os.path.exists(MODEL_PATH):
    print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
    print("      Lütfen önce 'train_detector.py' scriptini çalıştırarak modeli eğitin.")
    exit()
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"   + Model '{MODEL_PATH}' başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenirken hata: {e}")
    exit()
print(">>> Yükleme tamamlandı.\n")
# --- Yükleme Sonu ---


# --- Tahmin Fonksiyonu ---
def predict_phishing(url_to_check):
    """Verilen URL için phishing tahmini yapar."""
    print(f"\n-> URL İşleniyor: {url_to_check}")

    # 1. Özellikleri Çıkar
    print("   -> Özellikler çıkarılıyor...")
    try:
        features_dict = extract_features(url_to_check)
        if not features_dict:
            print("   -> HATA: Bu URL için özellik çıkarılamadı (URL erişilemez veya sorunlu).")
            return "HATA: Özellik Çıkarılamadı", -1.0 # Hata durumu için -1 döndür
        features_df = pd.DataFrame([features_dict])
        print(f"   -> {len(features_df.columns)} özellik çıkarıldı.")
    except Exception as e:
        print(f"   -> HATA: Özellik çıkarma sırasında: {e}")
        return "HATA: Özellik Çıkarma", -1.0

    # 2. Özellik Sırasını ve Eksikleri Ayarla (Eğer isimler biliniyorsa)
    if FEATURE_NAMES:
        try:
            # Eksik sütunları NaN ile ekle
            for col in FEATURE_NAMES:
                if col not in features_df.columns:
                    features_df[col] = np.nan
            # Sütunları doğru sıraya koy
            features_df = features_df[FEATURE_NAMES]
        except Exception as e:
             print(f"   -> HATA: Özellik sütunları ayarlanırken: {e}")
             return "HATA: Özellik Eşleştirme", -1.0

    # 3. Eksik Değerleri Doldur (Varsa)
    if features_df.isnull().sum().sum() > 0:
        print("   -> Uyarı: Özelliklerde NaN bulundu, 0 ile dolduruluyor.")
        features_df = features_df.fillna(0)

    # 4. Özellikleri Ölçekle
    print("   -> Özellikler ölçekleniyor...")
    try:
        scaled_features = SCALER.transform(features_df)
    except ValueError as e:
         print(f"   -> HATA: Ölçekleme hatası: {e}")
         print("      -> Olası neden: Özellik sayısı scaler ile uyuşmuyor.")
         return "HATA: Ölçekleme", -1.0
    except Exception as e:
         print(f"   -> HATA: Ölçekleme sırasında beklenmedik hata: {e}")
         return "HATA: Ölçekleme", -1.0

    # 5. Tahmin Yap
    print("   -> Model ile tahmin yapılıyor...")
    try:
        prediction_prob = MODEL.predict(scaled_features, verbose=0)[0][0]
    except Exception as e:
        print(f"   -> HATA: Model tahmini sırasında hata: {e}")
        return "HATA: Model Tahmini", -1.0

    # 6. Sonucu Yorumla
    threshold = 0.5
    if prediction_prob >= threshold:
        result_label = "Phishing"
    else:
        result_label = "Meşru (Legitimate)"

    print("   -> Tahmin tamamlandı.")
    return result_label, prediction_prob

# --- Ana Çalışma Döngüsü ---
if __name__ == "__main__":
    print("="*40)
    print("   Phishing URL Tespit Aracı v1.0")
    print("="*40)
    print("Model ve Scaler yüklendi. Hazır.")
    print("Çıkmak için 'exit' veya 'quit' yazın.")

    while True:
        # Kullanıcıdan URL al
        try:
            url_input = input("\nKontrol edilecek URL'yi girin: ").strip()
        except EOFError: # Bazı ortamlarda Ctrl+D ile çıkış için
            print("\nÇıkış yapılıyor...")
            break

        # Çıkış komutları
        if url_input.lower() in ['exit', 'quit', 'q', 'çık', 'çıkış']:
            print("Programdan çıkılıyor...")
            break

        # Boş giriş kontrolü
        if not url_input:
            continue

        # Basit protokol kontrolü ve ekleme
        if not url_input.startswith('http://') and not url_input.startswith('https://'):
             print("   -> Uyarı: URL 'http://' veya 'https://' ile başlamıyor, 'http://' ekleniyor.")
             url_input = 'http://' + url_input

        # Tahmin fonksiyonunu çağır
        label, probability = predict_phishing(url_input)

        # Sonucu kullanıcıya göster
        print("\n" + "-"*15 + " SONUÇ " + "-"*15)
        print(f" URL: {url_input}")
        if probability != -1.0: # Eğer hata oluşmadıysa
             print(f" TAHMİN: {label}")
             print(f" Phishing Olasılığı: {probability:.4f} (%{probability*100:.2f})")
        else: # Hata oluştuysa
             print(f" DURUM: Tahmin yapılamadı - {label}") # label burada hata mesajını içerir
        print("-"*37)