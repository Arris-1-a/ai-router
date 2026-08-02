<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# AI-yönlendirici
<p align="merkez">
  <b>Akıllı LLM API Yönlendirici</b><br>
  Akıllı rota · Yük dengeleme · Geri Dönüş · Maliyet optimizasyonu<br>
  Akıllı Yönlendirme · Yük dengeleme · Failover · Maliyet optimizasyonu
</p>
## Genel Bakış | Genel Bakış
**ai-router**, akıllı LLM API yönetimi için üretime hazır bir Python araç setidir. Birden fazla yapay zeka sağlayıcısı (OpenAI, Anthropic, DeepSeek, Google ve daha fazlası) arasında yönlendirme talepleri için birleşik bir arayüz sunar; otomatik yük dengeleme, yedekleme, maliyet optimizasyonu ve kapsamlı metrikler sunar.
**ai-router**, LLM API isteklerini akıllıca yönetmek için üretim seviyesinde bir Python araç setidir. Birden fazla yapay zeka sağlayıcısı (OpenAI, Anthropic, DeepSeek, Google vb.) arasında akıllı yönlendirmeyi destekleyen birleşik bir arayüz sunar; otomatik yük dengeleme, devre sistemi, maliyet optimizasyonu ve kapsamlı metrik izleme sunar.
## Mimari | 架构
BLOCK0
## Özellikler | 功能特性
### 🧠 Akıllı Yönlendirici | 智能路由器
- **Çoklu Sağlayıcı**: OpenAI, Anthropic, DeepSeek, Google Gemini, genişletilebilir
- **Stratejiler**: Round-robin, ağırlıklı, en düşük gecikmeli, en düşük maliyetli, anlamsal, uyarlanabilir
- **Önbellekleme**: LRU + anlamsal benzerlik dedup önbellek
- **Yedek**: Devre kesici, otomatik yeniden deneme, çok katmanlı geri dönüş zincirleri
- **Oran Sınırlaması**: Token bucket algoritması, sağlayıcı başına ve küresel sınırlar
- **Metrikler**: Gecikme, maliyet, başarı oranı uyarılarla takip
### 📚 RAG Boru Hattı | RAG 管线








- **Chunking**: Sabit boyut, cümle, paragraf, özyinelemeli, markdown, kaydırmalı pencere
- **Gömülme**: OpenAI, cümle dönüştürücüleri, genişletilebilir arka uçlar
- **Geri Getirme**: Hibrit (BM25 + vektör), RRF füzyon, ağırlıklı kombinasyon
- **Sıralama Yenileme**: Puan bazlı, MMR çeşitliliği, çapraz kodlayıcı, LLM jüri
### 🤖 Ajan Çerçevesi | Agent 框架
- **ReAct Ajan**: Akıl yürütme + Araç kullanımıyla hareket döngüsü
- **Araç Sistemi**: Dekoratör tabanlı kayıt, JSON Şeması, doğrulama
- **Orkestrasyon**: Dizisel, paralel, tartışma, yönetici-çalışan kalıpları
- **Bellek**: Kısa süreli, uzun vadeli, epizodik, çalışma hafızası.
### 📊 Değerlendirme | 评估
- **Puanlama**: BLEU, ROUGE-1/2/L, anlamsal benzerlik, F1, tam eşleşme
- **Benchmarking**: Gecikme, veri verimliliği, maliyet, yük altında başarı oranı
### 🌐 API Sunucusu | API 服务
- **FastAPI**: OpenAI uyumlu INLINE14, gömülemeler, RAG, ajan uç noktaları
- **Middleware**: Loglama, hız sınırı, istek kimliği, zamanlama, CORS
## Hızlı Başlangıç | 快速开始
### Kurulum | 安装
BLOK1
Ya da kaynaktan:
BLOCK2




















### Temel Kullanım | Temel kullanım
BLOCK3
### Stratejiyle Yönlendirme | 策略路由
BLOCK4
### RAG Boru Hattı | RAG 管线
BLOCK5
### CLI Kullanımı | Komut satırı kullanımı
BLOCK6
### API Sunucusu | API 服务
BLOCK7
Sonra API'yi çağırın:
BLOCK8
## Yapılandırma | 配置
Ortam değişkenlerini ayarlayın:
BLOCK9
## Gelişim | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Veya docker-compose ile:









BLOCK12
## Proje Yapısı | 项目结构
BLOCK13
## Lisans | Lisans
MIT Lisansı — bkz. [LICENSE](LICENSE) dosyası.
## Katkıda bulunuyoram | 贡献
Katkılarınız memnuniyetle karşılanır! Lütfen [CONTRIBUTING.md](CONTRIBUTING.md) numaralı rehbere bakınız.
Katkılarınız memnuniyetle karşılanır! Katkı yönergeleri için lütfen [CONTRIBUTING.md](CONTRIBUTING.md) sayfasına bakınız.