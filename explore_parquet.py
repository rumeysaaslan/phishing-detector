import pandas as pd

# Sanal ortamının aktif olduğundan ve pyarrow'un kurulu olduğundan emin ol
try:
    print("Training.parquet içeriği okunuyor...")
    df_train = pd.read_parquet('Training.parquet') # Dosya adının doğru olduğundan emin ol
    print("\nTraining.parquet - İlk 5 Satır:")
    print(df_train.head())
    print("\nTraining.parquet - Sütun Bilgileri:")
    df_train.info()
    print("\nTraining.parquet - Etiket (Label) Dağılımı (varsa):")
    if 'Label' in df_train.columns: # Etiket sütununun adını kontrol et, 'Label' olmayabilir
        print(df_train['Label'].value_counts())
    elif 'label' in df_train.columns:
        print(df_train['label'].value_counts())
    elif 'Result' in df_train.columns: # Veya başka bir isim olabilir
        print(df_train['Result'].value_counts())
    else:
        print("Uygun etiket sütunu bulunamadı, sütunları manuel kontrol edin.")


    print("\n---------------------------------------------------\n")

    print("Testing.parquet içeriği okunuyor...")
    df_test = pd.read_parquet('Testing.parquet') # Dosya adının doğru olduğundan emin ol
    print("\nTesting.parquet - İlk 5 Satır:")
    print(df_test.head())
    print("\nTesting.parquet - Sütun Bilgileri:")
    df_test.info()
    print("\nTesting.parquet - Etiket (Label) Dağılımı (varsa):")
    if 'Label' in df_test.columns:
        print(df_test['Label'].value_counts())
    elif 'label' in df_test.columns:
        print(df_test['label'].value_counts())
    elif 'Result' in df_test.columns:
        print(df_test['Result'].value_counts())
    else:
        print("Uygun etiket sütunu bulunamadı, sütunları manuel kontrol edin.")

except Exception as e:
    print(f"Dosya okunurken hata oluştu: {e}")
    print("Lütfen 'pyarrow' kütüphanesinin kurulu olduğundan ve dosya adlarının/yollarının doğru olduğundan emin olun.")