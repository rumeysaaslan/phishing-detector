import tensorflow as tf
from tensorflow.keras.models import Sequential # Katmanları sıralı eklemek için model türü
from tensorflow.keras.layers import Dense, Dropout # Kullanacağımız katman türleri
# Dense: Tam bağlantılı katman (MLP'nin temeli)
# Dropout: Overfitting'i (ezberlemeyi) önlemeye yardımcı olan bir katman


def create_mlp_model(input_dim):
    """
    Belirtilen giriş boyutuna sahip basit bir MLP modeli oluşturur ve derler.

    Args:
        input_dim (int): Modelin alacağı giriş özelliklerinin sayısı.

    Returns:
        tensorflow.keras.models.Sequential: Derlenmiş Keras modeli.
    """
    # Modeli başlatıyoruz (katmanları sırayla ekleyeceğiz)
    model = Sequential(name='Phishing_Detector_MLP') # Modele bir isim verelim

    # 1. Gizli Katman (Hidden Layer):
    #   - Dense: Tam bağlantılı katman.
    #   - 64: Bu katmandaki nöron (yapay sinir) sayısı (deneyerek bulunabilir).
    #   - activation='relu': Aktivasyon fonksiyonu. Negatif değerleri sıfırlar, pozitifleri korur. Yaygın ve etkilidir.
    #   - input_dim=input_dim: Bu *sadece ilk katmanda* belirtilir. Modelin kaç tane giriş alacağını söyler.
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='Hidden_Layer_1'))

    # Dropout Katmanı:
    #   - Overfitting'i azaltmak için kullanılır.
    #   - 0.5: Nöronların %50'sini eğitim sırasında rastgele "leyeceğimiz Sequential modelini başlatıyoruz
    model = Sequential(name="Phishing_Detection_MLP") # Modele bir isim verelim

    # Giriş Katmanı (Implicit) ve İlk Gizli Katman:
    # input_dim: Bu katmanın kaç tane giriş alacağını belirtir.
    # units: Bu katmanda kaç tane nöron (yapay sinir hücresi) olacağı.
    # activation='relu': Rectified Linear Unit aktivasyon fonksiyonu. Negatif değerleri sıfırlar, pozitifleri korur.
    #                  Gizli katmanlar için yaygın ve genellikle iyi çalışan bir seçenektir.
    model.add(Dense(units=64, activation='relu', input_dim=input_dim, name='Hidden_Layer_1'))

    # Dropout Katmanı (Overfitting'i azaltmaya yardımcı olmak için):
    # rate=0.3: Eğitim sırasında rastgele olarak nöronların %30'unu devre dışı bırakır.
    # Bu, modelin tek bir özelliğe veya nörona aşırı bağımlı olmasını engeller.
    model.add(Dropout(rate=0.3, name='Dropout_1'))

    # İkinci Gizli Katman (Daha az nöronla):
    # Modelin daha karmaşık örüntüleri öğrenmesine yardımcı olabilir.
    model.add(Dense(units=32, activation='relu', name='Hidden_Layer_2'))
    model.add(Dropout(rate=0.2, name='Dropout_2')) # Biraz daha dropout

    # Çıkış Katmanı:
    # units=1: Çünkü ikili sınıflandırma yapıyoruz (Phishing=1, Legitimate=0). Tek bir çıkış yeterli.
    # activation='sigmoid': Sigmoid fonksiyonu, çıktıyı 0 ile 1 arasına sıkıştırır.
    #                     Bu çıktı, genellikle bir olasılık olarak yorumlanabilir (örneğin, phishing olma olasılığı).
    model.add(Dense(units=1, activation='sigmoid', name='Output_Layer'))

    # Modelin özetini ekrana yazdıralım (katmanları ve parametre sayılarını görmek için)
    print("\nModel Özeti:")
    model.summary()

    

    return model



if __name__ == "__main__":
    # Bu blok sadece dosya doğrudan çalıştırıldığında çalışır
    print("--- Test Modeli Oluşturuluyor ---")
    # Örnek bir giriş boyutu verelim (örneğin 19 özellik)
    # Gerçek boyut sonraki adımda belli olacak
    example_input_dimension = 19
    test_model = create_mlp_model(input_dim=example_input_dimension)
    print("--- Test Modeli Oluşturma Tamamlandı ---")

# --- Bu dosya doğrudan çalıştırıldığında bir şey yapmaması için ---
# Eğer istersen, test için buraya küçük bir kod ekleyebiliriz:
# if __name__ == "__main__":
#     # Örnek: 19 özellikle bir model oluştur
#     example_input_dim = 19
#     test_model = create_model(example_input_dim)
#     print("\nTest modeli başarıyla oluşturuldu.")
#     # Not: Bu model henüz derlenmedi veya eğitilmedi.
