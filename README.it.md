<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# AI-router
<p align="center">
  <b>Router Smart LLM API</b><br>
  Routing intelligente · Bilanciamento del carico · Riserva · Ottimizzazione <br>dei costi
  Routing intelligente · Bilanciamento del carico · Failover · Ottimizzazione dei costi
</p>
## Panoramica | Panoramica
**ai-router** è un toolkit Python pronto per la produzione per la gestione intelligente delle API LLM. Fornisce un'interfaccia unificata per instradare le richieste tra diversi provider di IA (OpenAI, Anthropic, DeepSeek, Google e altri) con bilanciamento automatico del carico, fallback, ottimizzazione dei costi e metriche complete.
**ai-router** è un toolkit Python di livello per la produzione per gestire in modo intelligente le richieste API LLM. Fornisce un'interfaccia unificata che supporta un routing intelligente tra più provider di IA (OpenAI, Anthropic, DeepSeek, Google, ecc.), con bilanciamento automatico del carico, failover, ottimizzazione dei costi e monitoraggio completo delle metriche.
## Architettura | 架构
BLOCK0
## Caratteristiche | 功能特性
### 🧠 Router Intelligente | 智能路由器
- **Multi-Provider**: OpenAI, Anthropic, DeepSeek, Google Gemini, estensibile
- **Strategie**: Round-robin, ponderato, latenza più bassa, costo più basso, semantica, adattivo
- **Cache**: LRU + cache semantica di dedup con similarità
- **Fallback**: interruttore automatico, ritenti automatici, catene di fallback multilivello
- **Limite di Velocità**: algoritmo del bucket dei token, per provider e limiti globali
- **Metriche**: Monitoraggio di latenza, costi e tasso di successo con avvisi
### 📚 Oleodotto RAG | RAG 管线








- **Slump**: Dimensione fissa, frase, paragrafo, ricorsivo, markdown, finestra scorrevole
- **Embedding**: OpenAI, trasformatori di frase, backend estensionali
- **Recupero**: Ibrido (BM25 + vettore), fusione RRF, combinazione pesata
- **Riclassifica**: basato sul punteggio, diversità MMR, cross-encoder, giudice LLM
### 🤖 Quadro Agenti | Agent 框架
- **ReAct Agent**: Ragionamento + Ciclo di azione con uso di strumenti
- **Sistema Strumenti**: registrazione basata su decoratori, schema JSON, validazione
- **Orchestrazione**: Pattern sequenziali, paralleli, di dibattito, manager-lavoratore
- **Memoria**: Memoria di lavoro a breve e lungo termine, episodico,
### 📊 Valutazione | 评估
- **Punteggio**: BLEU, ROUGE-1/2/L, somiglianza semantica, F1, corrispondenza esatta
- **Benchmarking**: Latenza, throughput, costo, tasso di successo sotto carico
### 🌐 API Server | API 服务
- **FastAPI**: INLINE14 compatibili OpenAI, embedding, RAG, endpoint agenti
- **Middleware**: Loging, limitazione di velocità, ID richiesta, tempistica, CORS
## Avvio rapido | 快速开始
### Installazione | 安装
BLOCK1
O dalla fonte:
BLOCK2




















### Uso di base | Uso di base
BLOCK3
### Instradamento con Strategia | 策略路由
BLOCK4
### Oleodotto RAG | RAG 管线
BLOCK5
### Utilizzo CLI | Utilizzo da riga di comando
BLOCK6
### API Server | API 服务
BLOCK7
Poi chiama l'API:
BLOCK8
## Configurazione | 配置
Variabili ambientali di impostazione:
BLOCK9
## Sviluppo | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Oppure con docker-compose:









BLOCK12
## Struttura del progetto | 项目结构
BLOCK13
## Licenza | Licenza
Licenza MIT — vedi fascicolo [LICENZA](LICENZA).
## Contribuendo | 贡献
I contributi sono benvenuti! Si prega di consultare [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida.
I contributi sono benvenuti! Si prega di consultare [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida sui contributi.