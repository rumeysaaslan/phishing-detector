import pandas as pd
import numpy as np
import time
import os
import joblib # Scaler'ı kaydetmek/yüklemek için

# Makine Öğrenmesi ve Değerlendirme Kütüphaneleri
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Derin Öğrenme Kütüphaneleri
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Görselleştirme Kütüphaneleri
import matplotlib.pyplot as plt # pip install matplotlib
import seaborn as sns # pip install seaborn

# Kendi yazdığımız modüller
try:
    from features import extract_features # Kendi özellik çıkarıcımızı kullanacağız
    from model import create_mlp_model
except ImportError:
    print("HATA: 'features.py' veya 'model.py' dosyası 'detector_model' klasöründe bulunamadı.")
    print("      Lütfen bu dosyaların doğru yerde olduğundan emin olun.")
    exit()


# --- Ayarlar ve Sabitler ---
PROCESSED_DATA_DIR = 'processed_data'
X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_train_features.parquet')
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'y_train.parquet')
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_test_features.parquet')
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'y_test.parquet')
SCALER_PATH = os.path.join('detector_model', 'scaler.joblib')
MODEL_SAVE_PATH = os.path.join('detector_model', 'phishing_detector_model.keras')
HISTORY_PLOT_PATH = os.path.join('detector_model', 'training_history.png')
CM_PLOT_PATH = os.path.join('detector_model', 'confusion_matrix.png')

# --- KONTROL DEĞİŞKENLERİ ---
# True yaparsak kaydedilmiş özellikleri kullanmayı dener, False yaparsak her seferinde çıkarır.
USE_CACHED_FEATURES = True
# True yaparsak veri setinin küçük bir örneğiyle çalışır (hızlı test için).
ENABLE_SAMPLING = False # <<<--- HIZLI TEST İÇİN True BIRAKABİLİRSİN ŞİMDİLİK
SAMPLE_SIZE_TRAIN = 500 # Eğer ENABLE_SAMPLING True ise kullanılacak eğitim örnek sayısı
SAMPLE_SIZE_TEST = 200  # Eğer ENABLE_SAMPLING True ise kullanılacak test örnek sayısı
# Özellik çıkarma sırasında URL'ler arasına bekleme ekle (engellenmemek için)?
ADD_SLEEP_DURING_EXTRACTION = False
SLEEP_TIME = 0.1 # Saniye cinsinden bekleme süresi (eğer üstteki True ise)
# --- KONTROL DEĞİŞKENLERİ SONU ---

# TensorFlow log seviyesi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# processed_data klasörü yoksa oluştur
if not os.path.exists(PROCESSED_DATA_DIR):
    try:
        print(f"'{PROCESSED_DATA_DIR}' klasörü oluşturuluyor...")
        os.makedirs(PROCESSED_DATA_DIR)
    except OSError as e:
         print(f"HATA: '{PROCESSED_DATA_DIR}' klasörü oluşturulamadı: {e}")


# --- Adım 1: Veri Seti Yükleme ve Temel Hazırlık ---
print("*"*20 + " Adım 1: Veri Setleri Yükleniyor (Parquet) " + "*"*20)
train_path = 'Training.parquet'
test_path = 'Testing.parquet'

try:
    df_train_raw = pd.read_parquet(train_path, columns=['url', 'status'])
    df_test_raw = pd.read_parquet(test_path, columns=['url', 'status'])
    print(f"'{train_path}' ({len(df_train_raw)} satır) ve '{test_path}' ({len(df_test_raw)} satır) başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"HATA: Parquet dosyası bulunamadı: {e}.")
    print("      Lütfen Kaggle'dan indirdiğiniz 'Training.parquet' ve 'Testing.parquet' dosyalarının")
    print(f"      bu script'in çalıştığı ana klasörde ({os.getcwd()}) olduğundan emin olun.")
    print("      Ayrıca 'pyarrow' kütüphanesinin kurulu olduğundan emin olun (`pip install pyarrow`).")
    exit()
except Exception as e:
    print(f"HATA: Parquet dosyası okunurken bir hata oluştu: {e}")
    exit()

# Etiketleri 0 ve 1'e dönüştürme ('phishing' -> 1, 'legitimate' -> 0)
print("\nAdım 1.5: Etiketler Dönüştürülüyor...")
try:
    df_train_raw['label'] = df_train_raw['status'].apply(lambda x: 1 if x == 'phishing' else 0)
    df_test_raw['label'] = df_test_raw['status'].apply(lambda x: 1 if x == 'phishing' else 0)
    df_train = df_train_raw[['url', 'label']].copy()
    df_test = df_test_raw[['url', 'label']].copy()
except KeyError:
    print("HATA: Parquet dosyalarında 'url' veya 'status' sütunu bulunamadı.")
    exit()
except Exception as e:
    print(f"HATA: Etiketler dönüştürülürken hata oluştu: {e}")
    exit()


# --- GELİŞTİRME/TEST İÇİN VERİ SETİNİ KÜÇÜLTME (OPSİYONEL) ---
if ENABLE_SAMPLING:
    print("\n" + "*"*10 + " UYARI: Veri seti geliştirme amacıyla küçültülüyor! " + "*"*10)
    if len(df_train) > SAMPLE_SIZE_TRAIN:
        # Stratified sampling yapmak daha iyi olabilir ama şimdilik random
        df_train = df_train.sample(n=SAMPLE_SIZE_TRAIN, random_state=42).reset_index(drop=True)
        print(f"   -> Eğitim seti rastgele {len(df_train)} örneğe düşürüldü.")
    else:
        print(f"   -> Eğitim seti zaten {len(df_train)} örnek içeriyor, küçültme yapılmadı.")

    if len(df_test) > SAMPLE_SIZE_TEST:
        df_test = df_test.sample(n=SAMPLE_SIZE_TEST, random_state=42).reset_index(drop=True)
        print(f"   -> Test seti rastgele {len(df_test)} örneğe düşürüldü.")
    else:
        print(f"   -> Test seti zaten {len(df_test)} örnek içeriyor, küçültme yapılmadı.")
    print("*"*60)
# --- VERİ SETİ KÜÇÜLTME SONU ---


print("\nEğitim Seti (Potansiyel Olarak Küçültülmüş) Etiket Dağılımı:")
print(df_train['label'].value_counts())
print("\nTest Seti (Potansiyel Olarak Küçültülmüş) Etiket Dağılımı:")
print(df_test['label'].value_counts())


# --- Adım 2: Özellik Çıkarma (veya Önbellekten Yükleme) ---
print("\n" + "*"*20 + " Adım 2: Özellik Çıkarma / Yükleme " + "*"*20)

features_loaded_from_cache = False
# Önce önbellekten yüklemeyi dene
if USE_CACHED_FEATURES and os.path.exists(X_TRAIN_PATH) and os.path.exists(Y_TRAIN_PATH) \
   and os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
    print(f"-> Kaydedilmiş özellikler '{PROCESSED_DATA_DIR}' klasöründe bulundu.")
    print("-> Önbellekten yükleniyor...")
    try:
        X_train_features = pd.read_parquet(X_TRAIN_PATH)
        y_train = pd.read_parquet(Y_TRAIN_PATH)['label'].values
        X_test_features = pd.read_parquet(X_TEST_PATH)
        y_test = pd.read_parquet(Y_TEST_PATH)['label'].values
        print("   -> Kaydedilmiş özellikler ve etiketler başarıyla yüklendi.")
        features_loaded_from_cache = True

        # Cache'den yüklenen veri ile sampling sonrası df boyutları uyuşmuyorsa UYAR VE GEREKİRSE AYARLA
        if ENABLE_SAMPLING and (len(X_train_features) > SAMPLE_SIZE_TRAIN or len(X_test_features) > SAMPLE_SIZE_TEST):
             print("   -> DİKKAT: Önbellekten yüklenen örnek sayısı ({}+{}) mevcut örnekleme".format(len(X_train_features), len(X_test_features)))
             print("             ayarlarından ({}+{}) büyük. Önbellek tam veriye ait olabilir.".format(SAMPLE_SIZE_TRAIN, SAMPLE_SIZE_TEST))
             print("             Önbellekteki verinin örneklemesi yapılıyor...")
             # Cache'den yüklenen veriyi de sample edelim ki boyutlar tutsun
             if len(X_train_features) > SAMPLE_SIZE_TRAIN:
                 train_indices_to_keep = np.random.RandomState(seed=42).choice(X_train_features.index, size=SAMPLE_SIZE_TRAIN, replace=False)
                 X_train_features = X_train_features.loc[train_indices_to_keep].reset_index(drop=True)
                 y_train = y_train[train_indices_to_keep]
             if len(X_test_features) > SAMPLE_SIZE_TEST:
                 test_indices_to_keep = np.random.RandomState(seed=42).choice(X_test_features.index, size=SAMPLE_SIZE_TEST, replace=False)
                 X_test_features = X_test_features.loc[test_indices_to_keep].reset_index(drop=True)
                 y_test = y_test[test_indices_to_keep]
             print(f"   -> Önbellek verisi {len(X_train_features)} eğitim, {len(X_test_features)} test örneğine düşürüldü.")

    except Exception as e:
        print(f"   -> HATA: Kaydedilmiş özellikler yüklenemedi: {e}. Özellik çıkarma yeniden yapılacak.")
        features_loaded_from_cache = False

# Eğer önbellekten yüklenemediyse veya istenmediyse, özellikleri çıkar
if not features_loaded_from_cache:
    print("-> Önbellek kullanılmıyor veya bulunamadı. Özellik çıkarma başlatılıyor (Bu işlem uzun sürebilir!)...")

    # --- DÜZELTİLMİŞ FONKSİYON TANIMI ---
    def extract_features_for_df(df_input, df_original_for_labels, set_name="Bilinmeyen"):
        """DataFrame alıp her URL için özellik çıkarır ve yeni DataFrame döndürür."""
        all_features_list = []
        successful_indices_list = [] # Orijinal indexleri tutacak
        print(f"\n  -> {set_name} Seti: {len(df_input)} URL için özellik çıkarma başlıyor...")
        start_time_loop = time.time()
        # iterrows hem indeksi (df_input'un indeksi) hem de satırı verir
        for i, (index, row) in enumerate(df_input.iterrows()):
            original_df_index = df_input.index[i] # Bu index, df_input'un mevcut (sample edilmiş olabilir) indeksidir
                                                  # Etiketleri almak için df_original_for_labels'ın indeksine ihtiyacımız var.
                                                  # Eğer sample yapıldıysa, df_input'un indeksi resetlendiği için df_original_for_labels ile eşleşmeyebilir.
                                                  # En güvenlisi, başarılı URL'lerin etiketlerini döngü sonunda df_original_for_labels'dan almak.
            url = row['url']
            if (i + 1) % 100 == 0:
                elapsed_loop = time.time() - start_time_loop
                print(f"    -> {set_name} ({i + 1}/{len(df_input)}) işleniyor... (Geçen Süre: {elapsed_loop:.1f}s)")
            try:
                url_features = extract_features(url)
                if url_features:
                    all_features_list.append(url_features)
                    # Başarılı olanın df_input içindeki indeksini kaydedelim, sonra orijinal df'ten eşleşen etiketi alırız
                    successful_indices_list.append(original_df_index) # Başarılı olanın df_input'taki indeksini kaydet
                else:
                    print(f"    -> Uyarı: {url[:70]}... için özellik çıkarılamadı (boş döndü), atlanıyor.")
            except KeyboardInterrupt:
                 print("\nKullanıcı tarafından işlem durduruldu.")
                 exit()
            except Exception as e:
                print(f"    -> HATA: {url[:70]}... işlenirken beklenmedik hata: {e}")

            if ADD_SLEEP_DURING_EXTRACTION:
                time.sleep(SLEEP_TIME)

        end_time_loop = time.time()
        print(f"  -> {set_name} Seti: Özellik çıkarma tamamlandı. Süre: {end_time_loop - start_time_loop:.2f} saniye.")
        if not all_features_list:
            return None, None
        # DataFrame'i oluştururken başarılı df_input indekslerini kullanalım
        features_df_output = pd.DataFrame(all_features_list, index=successful_indices_list)
        # Başarılı indekslere karşılık gelen etiketleri al (orijinal df'ten)
        labels_output = df_original_for_labels.loc[successful_indices_list, 'label'].values
        return features_df_output, labels_output
    # --- FONKSİYON TANIMI SONU ---


    # --- DÜZELTİLMİŞ FONKSİYON ÇAĞRILARI ---
    # Eğitim verisi için çıkar (Orijinal df_train referans olarak verilir)
    X_train_features, y_train = extract_features_for_df(df_train, df_train_raw, "Eğitim")
    if X_train_features is None:
        print("HATA: Eğitim verisi için özellik çıkarılamadı. Script durduruluyor.")
        exit()

    # Test verisi için çıkar (Orijinal df_test referans olarak verilir)
    X_test_features, y_test = extract_features_for_df(df_test, df_test_raw, "Test")
    if X_test_features is None:
        print("HATA: Test verisi için özellik çıkarılamadı. Script durduruluyor.")
        exit()
    # --- DÜZELTİLMİŞ FONKSİYON ÇAĞRILARI SONU ---


    # Çıkarılan Özellikleri Kaydet
    print("\nAdım 2c: Çıkarılan Özellikler ve Etiketler Kaydediliyor...")
    try:
        # İndeksleri resetleyerek kaydedelim
        X_train_features.reset_index(drop=True).to_parquet(X_TRAIN_PATH)
        pd.DataFrame({'label': y_train}).to_parquet(Y_TRAIN_PATH)
        X_test_features.reset_index(drop=True).to_parquet(X_TEST_PATH)
        pd.DataFrame({'label': y_test}).to_parquet(Y_TEST_PATH)
        print(f"   -> Özellikler ve etiketler '{PROCESSED_DATA_DIR}' klasörüne başarıyla kaydedildi.")
    except Exception as e:
        print(f"   -> HATA: Özellikler kaydedilemedi: {e}")
        print("      -> Script devam edecek ancak bir sonraki çalıştırmada özellikler yeniden çıkarılacak.")


# Özellik DataFrame'lerinin bilgilerini yazdır
print("\nEğitim Özellikleri DataFrame'i Bilgileri:")
X_train_features.info()
print("\nTest Özellikleri DataFrame'i Bilgileri:")
X_test_features.info()

# Adım 3 için X_train ve X_test'i ayarla
X_train = X_train_features.copy()
X_test = X_test_features.copy()


# --- Adım 3: Veri Ön İşleme ---
print("\n" + "*"*20 + " Adım 3: Veri Ön İşleme " + "*"*20)

train_median = None
print(" - Eğitim Seti: Eksik değerler kontrol ediliyor ve dolduruluyor...")
if X_train.isnull().sum().sum() > 0:
    print(f"   -> NaN değerler bulundu (Eğitim): \n{X_train.isnull().sum()[X_train.isnull().sum() > 0]}")
    try:
        numeric_cols_train = X_train.select_dtypes(include=np.number).columns
        train_median = X_train[numeric_cols_train].median()
        X_train[numeric_cols_train] = X_train[numeric_cols_train].fillna(train_median)
        print("   -> Sayısal NaN değerler eğitim seti medyanları ile dolduruldu.")
    except Exception as e:
        print(f"   -> HATA: Eğitim seti NaN doldurma sırasında hata: {e}")
else:
    print("   -> Eğitim setinde eksik değer bulunamadı.")

print(" - Test Seti: Eksik değerler kontrol ediliyor ve eğitim medyanı ile dolduruluyor...")
if X_test.isnull().sum().sum() > 0:
     print(f"   -> NaN değerler bulundu (Test): \n{X_test.isnull().sum()[X_test.isnull().sum() > 0]}")
     if train_median is not None:
         try:
            numeric_cols_test = X_test.select_dtypes(include=np.number).columns
            common_numeric_cols = numeric_cols_test.intersection(train_median.index)
            X_test[common_numeric_cols] = X_test[common_numeric_cols].fillna(train_median[common_numeric_cols])
            print("   -> Sayısal NaN değerler eğitim seti medyanları ile dolduruldu (Test).")
            if X_test.isnull().sum().sum() > 0:
                 print("   -> Kalan NaN değerler 0 ile dolduruluyor.")
                 X_test = X_test.fillna(0)
         except Exception as e:
            print(f"   -> HATA: Test seti NaN doldurma sırasında hata: {e}")
     else:
         print("   -> Uyarı: Eğitim setinde medyan hesaplanamadığı için Test seti NaN değerleri doldurulamadı. 0 ile dolduruluyor.")
         X_test = X_test.fillna(0)
else:
     print("   -> Test setinde eksik değer bulunamadı.")


print(" - Özellikler ölçekleniyor (StandardScaler)...")
scaler = StandardScaler()
print("   -> Scaler eğitim verisine fit ediliyor...")
# NaN içermeyen sütunlar üzerinden fit etme mantığı kaldırıldı, NaN'lar doldurulduğu varsayılıyor.
# Eğer hala NaN kalma ihtimali varsa, önceki kod daha güvenli olabilir.
scaler.fit(X_train)

print("   -> Eğitim verisi transform ediliyor...")
X_train_scaled = scaler.transform(X_train)
print("   -> Test verisi transform ediliyor...")
X_test_scaled = scaler.transform(X_test)
print("   -> Ölçekleme tamamlandı.")

print("   -> Scaler kaydediliyor...")
try:
    joblib.dump(scaler, SCALER_PATH)
    print(f"      -> Scaler başarıyla '{SCALER_PATH}' olarak kaydedildi.")
except Exception as e:
    print(f"      -> HATA: Scaler kaydedilemedi: {e}")

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nÖlçeklenmiş Eğitim Özellikleri İlk 5 Satır:")
print(X_train.head())

print("\n" + "*"*20 + " Adım 4: Veri Bölme (Atlandı) " + "*"*20)
print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
print(f"Test seti boyutu: {X_test.shape[0]} örnek")
print(f"Özellik sayısı: {X_train.shape[1]}")

print("\n" + "*"*20 + " Adım 5: Model Oluşturma ve Derleme " + "*"*20)
try:
    input_dimension = X_train.shape[1]
    model = create_mlp_model(input_dim=input_dimension)
    print("\n   -> Model başarıyla oluşturuldu.")

    print("\nAdım 5.5: Model Derleniyor...")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("   -> Model başarıyla derlendi.")
except Exception as e:
    print(f"HATA: Model oluşturma veya derleme sırasında hata: {e}")
    exit()

print("\n" + "*"*20 + " Adım 6: Model Eğitimi " + "*"*20)
start_time_train = time.time()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

print("Eğitim başlıyor...")
try:
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=1)
except Exception as e:
    print(f"HATA: Model eğitimi sırasında bir hata oluştu: {e}")
    exit()

end_time_train = time.time()
print(f"\nModel eğitimi tamamlandı. Süre: {end_time_train - start_time_train:.2f} saniye.")

print("\n" + "*"*18 + " Adım 6.5: Eğitim Grafikleri " + "*"*18)
if hasattr(history, 'history') and all(k in history.history for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss_hist = history.history['loss']
    val_loss_hist = history.history['val_loss']
    epochs_range = range(len(acc))

    try:
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Eğitim Doğruluğu', marker='.')
        plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu', marker='.')
        plt.legend(loc='lower right')
        plt.title('Eğitim ve Doğrulama Doğruluğu')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss_hist, label='Eğitim Kaybı', marker='.')
        plt.plot(epochs_range, val_loss_hist, label='Doğrulama Kaybı', marker='.')
        plt.legend(loc='upper right')
        plt.title('Eğitim ve Doğrulama Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(HISTORY_PLOT_PATH)
        print(f"   -> Eğitim grafikleri '{HISTORY_PLOT_PATH}' olarak kaydedildi.")
        # plt.show()
    except Exception as e_plt:
        print(f"   -> HATA: Eğitim grafikleri çizilirken/kaydedilirken hata: {e_plt}")
else:
    print("   -> Eğitim geçmişinde beklenen metrikler bulunamadı, grafikler çizilemiyor.")

print("\n" + "*"*18 + " Adım 7: Model Değerlendirme " + "*"*18)
print("-> Model Test Seti Üzerinde Değerlendiriliyor...")
try:
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Seti Sonuçları:")
    print(f"  Kayıp (Loss): {loss:.4f}")
    print(f"  Doğruluk (Accuracy): {accuracy:.4f} (yani %{accuracy*100:.2f})")
except Exception as e:
    print(f"HATA: Model değerlendirme sırasında hata: {e}")
    loss, accuracy = -1.0, -1.0

print("\n" + "*"*14 + " Adım 7.5: Detaylı Metrikler " + "*"*14)
if accuracy != -1.0:
    try:
        print("-> Test seti için tahminler yapılıyor...")
        y_pred_probabilities = model.predict(X_test, verbose=0)
        y_pred_classes = (y_pred_probabilities > 0.5).astype("int32").flatten()
        y_test_flat = y_test.flatten()

        print("\nKarışıklık Matrisi (Test Seti):")
        cm = confusion_matrix(y_test_flat, y_pred_classes)
        print(cm)

        print("-> Karışıklık matrisi grafiği oluşturuluyor...")
        try:
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Meşru (0)', 'Phishing (1)'],
                        yticklabels=['Meşru (0)', 'Phishing (1)'])
            plt.xlabel('Tahmin Edilen Etiket')
            plt.ylabel('Gerçek Etiket')
            plt.title('Karışıklık Matrisi')
            plt.savefig(CM_PLOT_PATH)
            print(f"   -> Karışıklık matrisi grafiği '{CM_PLOT_PATH}' olarak kaydedildi.")
            # plt.show()
        except Exception as e_plt:
            print(f"   -> HATA: Karışıklık matrisi grafiği çizilirken/kaydedilirken hata: {e_plt}")

        print("\nSınıflandırma Raporu (Test Seti):")
        report = classification_report(y_test_flat, y_pred_classes, target_names=['Meşru (0)', 'Phishing (1)'], zero_division=0)
        print(report)

    except Exception as e_report:
         print(f"   -> HATA: Detaylı metrikler oluşturulurken hata: {e_report}")
else:
    print("-> Model değerlendirme başarısız olduğu için detaylı metrikler oluşturulamadı.")

print("\n" + "*"*20 + " Adım 8: Model Kaydetme " + "*"*20)
if accuracy != -1.0:
    print("-> Eğitilmiş Model Kaydediliyor...")
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"   -> Model başarıyla '{MODEL_SAVE_PATH}' olarak kaydedildi.")
    except Exception as e:
        print(f"   -> HATA: Model kaydedilemedi: {e}")
else:
    print("-> Model eğitimi veya değerlendirmesi başarısız olduğu için model kaydedilmedi.")


print("\n" + "="*50)
print("--- train_detector.py script'i tamamlandı. ---")
print("="*50)