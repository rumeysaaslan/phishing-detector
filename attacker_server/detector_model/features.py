import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlsplit # URL'leri parçalamak için
import re # Regex işlemleri için (örn: IP adresi bulma)
import time # İstekler arasına küçük bekleme eklemek için (sunucuyu yormamak adına)
import pandas as pd # Veri işleme için (CSV okuma/yazma)

def get_html_content(url, timeout=5):
    """
    Verilen URL'den HTML içeriğini çekmeye çalışır.
    Hata durumunda None döner.
    """
    try:
        # Web sunucusuna istek gönder (timeout belirleyerek çok uzun beklemeyi önle)
        # verify=False ekledik çünkü yerel sunucumuzda HTTPS sertifikası yok
        # Gerçek dünyada HTTPS siteleri için verify=True olmalı veya dikkatli kullanılmalı.
        response = requests.get(url, timeout=timeout, verify=False)

        # HTTP status kodunu kontrol et (200 OK ise başarılı)
        if response.status_code == 200:
            # İçeriği UTF-8 olarak decode etmeye çalış, olmazsa response'un kendi tahminini kullan
            try:
                content = response.content.decode('utf-8')
            except UnicodeDecodeError:
                content = response.text # requests'in tahmin ettiği encoding'i kullan
            return content
        else:
            # Uyarı mesajındaki potansiyel hatalı karakter düzeltildi
            print(f"Uyarı: {url} için HTTP durum kodu {response.status_code} alındı.")
            return None
    except requests.exceptions.RequestException as e:
        # Bağlantı hatası, timeout vb.
        # Hata mesajındaki potansiyel hatalı karakter düzeltildi
        print(f"Hata: {url} adresine ulaşılamadı. {e}")
        return None
    except Exception as e:
        # Diğer beklenmedik hatalar
        print(f"Beklenmedik Hata ({url}): {e}")
        return None

# --- BU FONKSİYONUN GİRİNTİSİ DÜZELTİLDİ ---
def extract_features(url):
    """
    Bir URL ve onun HTML içeriğinden özellikler çıkarır.
    (Docstring düzgün kapatıldı)
    """
    # --- BU SATIRDAN İTİBAREN TÜM İÇERİK BİR SEVİYE İÇERİ GİRİNTİLENDİ ---
    features = {}
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    # --- URL Tabanlı Özellikler ---
    features['url_length'] = len(url)
    features['domain_length'] = len(domain)
    features['path_length'] = len(path)
    features['num_dots_url'] = url.count('.')
    features['num_hyphens_url'] = url.count('-')
    features['num_at_url'] = url.count('@')
    features['num_slash_url'] = url.count('/')
    features['num_query_params'] = len(parsed_url.query.split('&')) if parsed_url.query else 0
    # Alan adında IP adresi var mı? (Basit regex kontrolü)
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    features['domain_is_ip'] = 1 if re.search(ip_pattern, domain) else 0
    # HTTPS kullanılıyor mu? (Protokole bakar)
    features['use_https'] = 1 if parsed_url.scheme == 'https' else 0
    # URL'de 'login', 'secure', 'account', 'update', 'password', 'signin' gibi şüpheli kelimeler var mı?
    suspicious_keywords = ['login', 'secure', 'account', 'update', 'password', 'signin', 'verify', 'bank']
    features['contains_suspicious_keyword'] = 1 if any(keyword in url.lower() for keyword in suspicious_keywords) else 0

    # --- HTML Tabanlı Özellikler ---
    html_content = get_html_content(url)
    # Başlangıçta HTML'den çıkarılacak özelliklere varsayılan değerler verelim (eğer HTML alınamazsa)
    features['html_has_form'] = 0
    features['html_has_password_field'] = 0
    features['html_num_links'] = 0
    features['html_num_script_tags'] = 0
    features['html_content_length'] = 0 # İçerik alınamazsa 0
    features['html_has_iframe'] = 0 # iframe için de varsayılan eklendi
    features['title_contains_suspicious_keyword'] = 0 # başlık için de varsayılan eklendi

    if html_content:
        features['html_content_length'] = len(html_content)
        try: # BeautifulSoup bazen hatalı HTML'de sorun çıkarabilir, try-except eklemek iyi olur
            soup = BeautifulSoup(html_content, 'lxml') # lxml parser daha hızlı, yoksa 'html.parser' kullanır

            # Form etiketi var mı?
            forms = soup.find_all('form')
            features['html_has_form'] = 1 if forms else 0

            # Şifre giriş alanı (<input type="password">) var mı?
            password_inputs = soup.find_all('input', attrs={'type': 'password'})
            features['html_has_password_field'] = 1 if password_inputs else 0

            # Toplam link sayısı (<a> etiketleri)
            links = soup.find_all('a')
            features['html_num_links'] = len(links)

            # Script etiketi sayısı (<script>)
            scripts = soup.find_all('script')
            features['html_num_script_tags'] = len(scripts)

            # Iframe var mı?
            iframes = soup.find_all('iframe')
            features['html_has_iframe'] = 1 if iframes else 0

            # Sayfa başlığında (<title>) şüpheli kelime var mı?
            title_tag = soup.find('title')
            # features['title_contains_suspicious_keyword'] = 0 # Bu satır yukarı taşındı
            if title_tag and title_tag.string:
               if any(keyword in title_tag.string.lower() for keyword in suspicious_keywords):
                   features['title_contains_suspicious_keyword'] = 1
        except Exception as e:
            print(f"Hata: BeautifulSoup HTML ayrıştırırken sorun oluştu ({url}). {e}")
            # Hata durumunda HTML özelliklerini varsayılan değerlerde bırakırız

    # Küçük bir bekleme ekleyelim (sunucuyu çok hızlı sorgulamamak için)
    # time.sleep(0.1) # Çok fazla URL varsa bu süreyi ayarlayabilir veya kaldırabilirsin

    return features