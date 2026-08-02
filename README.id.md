<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# ai-router
<p align="center">
  <b>Router API LLM Pintar</b><br>
  Routing cerdas · Penyeimbangan beban · Cadangan · Optimasi <br>biaya
  Routing Pintar · Penyeimbangan beban · Failover · Optimasi biaya
</p>
## Ikhtisar | Ikhtisar
**ai-router** adalah toolkit Python siap produksi untuk manajemen API LLM yang cerdas. Perangkat ini menyediakan antarmuka terpadu untuk mengarahkan permintaan di berbagai penyedia AI (OpenAI, Anthropic, DeepSeek, Google, dan lainnya) dengan penyeimbangan beban otomatis, cadangan, optimasi biaya, dan metrik komprehensif.
**ai-router** adalah toolkit Python kelas produksi untuk mengelola permintaan API LLM secara cerdas. Perangkat ini menyediakan antarmuka terpadu yang mendukung routing cerdas di berbagai penyedia AI (OpenAI, Anthropic, DeepSeek, Google, dll.), dengan fitur load balancing otomatis, failover, optimasi biaya, dan pemantauan metrik yang komprehensif.
## Arsitektur | 架构
BLOK0
## Fitur | 功能特性
### 🧠 Router Pintar | 智能路由器
- **Multi-Provider**: OpenAI, Anthropic, DeepSeek, Google Gemini, dapat diperluas
- **Strategi**: Round-robin, berbobot, latensi terendah, biaya terendah, semantik, adaptif
- **Caching**: LRU + cache dedup kesamaan semantik
- **Fallback**: Pemutus sirkuit, coba ulang otomatis, rantai fallback multi-level
- **Pembatasan Rate**: Algoritma bucket token, batas per penyedia & global
- **Metrics**: Latensi, biaya, pelacakan tingkat keberhasilan dengan peringatan
### 📚 Pipa RAG | RAG 管线








- **Chunking**: Ukuran tetap, kalimat, paragraf, rekursif, markdown, jendela geser
- **Penyematan**: OpenAI, transformer kalimat, backend yang dapat diperluas
- **Pengambilan**: Hibrida (BM25 + vektor), fusi RRF, kombinasi berbobot
- **Reranking**: Berdasarkan skor, keragaman MMR, cross-encoder, juri LLM
### 🤖 Kerangka Agen | Agen 框架
- **Agen ReAct**: Penalaran + Loop Bertindak dengan penggunaan alat
- **Sistem Alat**: Pendaftaran berbasis dekorator, skema JSON, validasi
- **Orkestrasi**: Pola berurutan, paralel, debat, manajer-pekerja
- **Memori**: Memori kerja jangka pendek, jangka panjang, episodik,
### 📊 Evaluasi | 评估
- **Skor**: BLEU, ROUGE-1/2/L, kesamaan semantik, F1, kecocokan tepat
- **Benchmarking**: Latensi, throughput, biaya, tingkat keberhasilan di bawah beban
### 🌐 Server API | API 服务
- **FastAPI**: INLINE14, embedding, RAG, endpoint agen yang kompatibel dengan OpenAI
- **Middleware**: Pencatatan, pembatasan rate, ID permintaan, waktu, CORS
## Mulai Cepat | 快速开始
### Instalasi | 安装
BLOK1
Atau dari sumber:
BLOK2




















### Penggunaan Dasar | Penggunaan dasar
BLOK3
### Routing dengan Strategi | 策略路由
BLOK4
### Pipa RAG | RAG 管线
BLOK5
### Penggunaan CLI | Penggunaan baris perintah
BLOK6
### Server API | API 服务
BLOK7
Kemudian panggil API:
BLOK8
## Konfigurasi | 配置
Atur variabel lingkungan:
BLOK9
## Pengembangan | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Atau dengan docker-compose:









BLOCK12
## Struktur Proyek | 项目结构
BLOCK13
## Lisensi | Lisensi
Lisensi MIT — lihat file [LICENSE](LICENSE).
## Berkontribusi | 贡献
Kontribusi sangat diterima! Silakan lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk pedoman.
Kontribusi sangat diterima! Silakan merujuk ke [CONTRIBUTING.md](CONTRIBUTING.md) untuk pedoman kontribusi.