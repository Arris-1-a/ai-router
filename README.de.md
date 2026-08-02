<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# KI-Router
<p align="center">
  <b>Smart LLM API Router</b><br>
  Intelligente Routing · Lastverteilung · Rückzug · Kostenoptimierung<br>
  Smart Routing · Lastverteilung · Failover · Kostenoptimierung
</p>
## Überblick | Überblick
**ai-router** ist ein produktionsbereites Python-Toolkit für intelligentes LLM-API-Management. Es bietet eine einheitliche Schnittstelle für die Weiterleitung von Anfragen über mehrere KI-Anbieter (OpenAI, Anthropic, DeepSeek, Google und mehr) mit automatischer Lastverteilung, Rückfall, Kostenoptimierung und umfassenden Kennzahlen.
**ai-router** ist ein produktionsfähiges Python-Toolkit zur intelligenten Verwaltung von LLM-API-Anfragen. Es bietet eine einheitliche Schnittstelle, die intelligentes Routing über mehrere KI-Anbieter (OpenAI, Anthropic, DeepSeek, Google usw.) unterstützt, mit automatischer Lastverteilung, Failover, Kostenoptimierung und umfassender Kennzahlenüberwachung.
## Architektur | 架构
BLOCK0
## Merkmale | 功能特性
### 🧠 Smarter Router | 智能路由器
- **Multi-Provider**: OpenAI, Anthropic, DeepSeek, Google Gemini, erweiterbar
- **Strategien**: Round-Robin, gewichtet, niedrigste Latenz, kostengünstigste, semantisch, adaptiv
- **Caching**: LRU + semantischer Ähnlichkeits-Dedup-Cache
- **Rückfall**: Leistungsschalter, automatischer Wiederversuch, mehrstufige Rückfallketten
- **Rate Limiting**: Token-Bucket-Algorithmus, pro Anbieter und globale Limits
- **Kennzahlen**: Latenz-, Kosten- und Erfolgsratenverfolgung mit Alerts
### 📚 RAG-Pipeline | RAG 管线








- **Chunking**: Feste Größe, Satz, Absatz, rekursiv, Markdown, gleitendes Fenster
- **Embedding**: OpenAI, Satztransformatoren, erweiterbare Backends
- **Retrieval**: Hybrid (BM25+ Vektor), RRF-Fusion, gewichtete Kombination
- **Reranking**: Punktbasiert, MMR-Diversität, Cross-Encoder, LLM-Richter
### 🤖 Agenten-Framework | Agent 框架
- **ReAct Agent**: Schlussfolgerung + Handlungsschleife mit Werkzeugnutzung
- **Werkzeugsystem**: Dekoratorbasierte Registrierung, JSON-Schema, Validierung
- **Orchestrierung**: Sequenzielle, parallele, debattierende, Manager-Mitarbeiter-Muster
- **Gedächtnis**: Kurzzeit-, Langzeit-, episodisches Arbeitsgedächtnis
### 📊 Bewertung | 评估
- **Wertung**: BLEU, ROUGE-1/2/L, semantische Ähnlichkeit, F1, exakte Übereinstimmung
- **Benchmarking**: Latenz, Durchsatz, Kosten, Erfolgsrate unter Last
### 🌐 API-Server | API 服务
- **FastAPI**: OpenAI-kompatible INLINE14, Embeddings, RAG, Agentenendpunkte
- **Middleware**: Protokollierung, Rate-Limiting, Anfrage-ID, Timing, CORS
## Schneller Start | 快速开始
### Installation | 安装
BLOCK1
Oder aus der Quelle:
BLOCK2




















### Grundgebrauch | Grundlegende Verwendung
BLOCK3
### Routing mit Strategie | 策略路由
BLOCK4
### RAG-Pipeline | RAG 管线
BLOCK5
### CLI-Nutzung | Verwendung der Kommandozeile
BLOCK6
### API-Server | API 服务
BLOCK7
Dann rufen Sie die API auf:
BLOCK8
## Konfiguration | 配置
Setze Umweltvariablen:
BLOCK9
## Entwicklung | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Oder mit docker-compose:









BLOCK12
## Projektstruktur | 项目结构
BLOCK13
## Lizenz | Lizenz
MIT-Lizenz — siehe [LICENSE](LICENSE)-Datei.
## Beitrag | 贡献
Beiträge sind willkommen! Bitte siehe [CONTRIBUTING.md](CONTRIBUTING.md) für Richtlinien.
Beiträge sind willkommen! Bitte beachten Sie [CONTRIBUTING.md](CONTRIBUTING.md) für die Beitragsrichtlinien.