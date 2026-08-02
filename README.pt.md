<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# Roteador de IA
<p align="center">
  <b>Roteador inteligente de API LLM</b><br>
  Roteamento inteligente · Balanceamento de carga · Reserva · Otimização <br>de custos
  Roteamento Inteligente · Balanceamento de carga · Failover · Otimização de custos
</p>
## Visão Geral | Visão geral
**ai-router** é um kit de ferramentas Python pronto para produção para gerenciamento inteligente de APIs LLM. Ele oferece uma interface unificada para roteamento de requisições entre múltiplos provedores de IA (OpenAI, Anthropic, DeepSeek, Google e outros) com balanceamento automático de carga, recurso de backup, otimização de custos e métricas abrangentes.
**ai-router** é um kit de ferramentas Python de nível de produção para gerenciar de forma inteligente requisições de API LLM. Ele oferece uma interface unificada que suporta roteamento inteligente entre múltiplos provedores de IA (OpenAI, Anthropic, DeepSeek, Google, etc.), com balanceamento automático de carga, failover, otimização de custos e monitoramento abrangente de métricas.
## Arquitetura | 架构
BLOCK0
## Características | 功能特性
### 🧠 Roteador Inteligente | 智能路由器
- **Multi-Provedor**: OpenAI, Anthropic, DeepSeek, Google Gemini, extensível
- **Estratégias**: Round-robin, ponderado, menor latência, menor custo, semântico, adaptativo
- **Cache**: LRU + cache semântica de similaridade dedup
- **Reserva de recurso**: Disjuntor, retentativa automática, cadeias de recuo em múltiplos níveis
- **Limitação de Taxa**: Algoritmo bucket de tokens, por provedor e limites globais
- **Métricas**: Rastreamento de latência, custo e taxa de sucesso com alertas
### 📚 Oleoduto RAG | RAG 管线








- **Chunk*: tamanho fixo, frase, parágrafo, recursivo, markdown, janela deslizante
- **Embedding**: OpenAI, transformadores de sentença, backends extensíveis
- **Recuperação**: Híbrido (BM25 + vetor), fusão RRF, combinação ponderada
- **Reclassificação**: Baseado em pontuação, diversidade MMR, cross-encoder, juiz LLM
### 🤖 Estrutura de Agentes | Agent 框架
- **ReAct Agent**: Loop de raciocínio + atuação com uso de ferramentas
- **Sistema de Ferramentas**: Registro baseado em decorador, Esquema JSON, validação
- **Orquestração**: Padrões sequenciais, paralelos, de debate, gerente-trabalhador
- **Memória**: Memória de trabalho de curto prazo, longo prazo, episódica
### 📊 Avaliação | 评估
- **Pontuação**: BLEU, ROUGE-1/2/L, similaridade semântica, F1, correspondência exata
- **Benchmarking**: Latência, throughput, custo, taxa de sucesso sob carga
### 🌐 API Server | API 服务
- **FastAPI**: INLINE14 compatíveis com OpenAI, embeddings, RAG, endpoints de agentes
- **Middleware**: Log, limitação de taxa, ID de requisição, temporização, CORS
## Início Rápido | 快速开始
### Instalação | 安装
BLOCK1
Ou da fonte:
BLOCK2




















### Uso Básico | Uso básico
BLOCK3
### Roteamento com Estratégia | 策略路由
BLOCK4
### Oleoduto RAG | RAG 管线
BLOCK5
### Uso da CLI | Uso na linha de comando
BLOCK6
### API Server | API 服务
BLOCK7
Depois, ligue para a API:
BLOCK8
## Configuração | 配置
Defina variáveis do ambiente:
BLOCK9
## Desenvolvimento | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Ou com docker-compose:









BLOCK12
## Estrutura do Projeto | 项目结构
BLOCK13
## Licença | Licença
Licença MIT — veja o arquivo [LICENÇA](LICENÇA).
## Contribuindo | 贡献
Contribuições são bem-vindas! Por favor, consulte [CONTRIBUTING.md](CONTRIBUTING.md) para orientações.
Contribuições são bem-vindas! Por favor, consulte [CONTRIBUTING.md](CONTRIBUTING.md) para as diretrizes de contribuição.