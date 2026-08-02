<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# Router IA
<p align="center">
  <b>Router inteligente de API LLM</b><br>
  Enrutamiento inteligente · Balanceo de carga · Respaldo · Optimización <br>de costes
  Enrutamiento inteligente · Balanceo de carga · Conmutación por error · Optimización de costes
</p>
## Resumen | Resumen
**AI-router** es un kit de herramientas Python listo para producción para la gestión inteligente de APIs LLM. Proporciona una interfaz unificada para enrutar solicitudes entre múltiples proveedores de IA (OpenAI, Anthropic, DeepSeek, Google y más) con balanceo automático de carga, respaldo, optimización de costes y métricas completas.
**ai-router** es un kit de herramientas de Python de grado de producción para gestionar de forma inteligente las solicitudes de la API de los LLM. Proporciona una interfaz unificada que soporta enrutamiento inteligente entre múltiples proveedores de IA (OpenAI, Anthropic, DeepSeek, Google, etc.), con balanceo automático de carga, conmutación por conmutación por error, optimización de costes y monitorización integral de métricas.
## Arquitectura | 架构
BLOQUE 0
## Características | 功能特性
### 🧠 Router inteligente | 智能路由器
- **Multi-Proveedor**: OpenAI, Anthropic, DeepSeek, Google Gemini, extensible
- **Estrategias**: Round-robin, ponderado, menor latencia, menor coste, semántico, adaptativo
- **Caché**: LRU + caché semántica de similitud deduplicada
- **Respaldo**: Interruptor automático, reintento automático, cadenas de respaldo multinivel
- **Limitación de tasa**: Algoritmo de bucket de token, por proveedor y límites globales
- **Métricas**: seguimiento de latencia, coste y tasa de éxito con alertas
### 📚 Oleoducto RAG | RAG 管线








- **Fragmentación**: Tamaño fijo, oración, párrafo, recursivo, markdown, ventana deslizante
- **Embedding**: OpenAI, transformadores de oración, backends extensibles
- **Recuperación**: Híbrido (BM25 + vector), fusión RRF, combinación ponderada
- **Reclasificación**: Basado en puntuación, diversidad MMR, cross-encoder, juez LLM
### 🤖 Marco de Agentes | Agent 框架
- **Agente ReAct**: Razonamiento + bucle de acción con uso de herramientas
- **Sistema de herramientas**: registro basado en decoradores, esquema JSON, validación
- **Orquestación**: Patrones secuenciales, paralelos, de debate, gerente-trabajador
- **Memoria**: Memoria de trabajo a corto y largo plazo, episódica
### 📊 Evaluación | 评估
- **Puntuación**: BLEU, ROUGE-1/2/L, similitud semántica, F1, coincidencia exacta
- **Benchmarking**: Latencia, rendimiento, coste, tasa de éxito bajo carga
### 🌐 API Server | API 服务
- **FastAPI**: INLINE14 compatibles con OpenAI, embeddings, RAG, endpoints de agentes
- **Middleware**: Registro, limitación de velocidad, ID de solicitudes, temporización, CORS
## Inicio rápido | 快速开始
### Instalación | 安装
BLOQUE1
O de la fuente:
BLOQUE2




















### Uso básico | Uso básico
BLOQUE3
### Enrutamiento con Estrategia | 策略路由
BLOCK4
### Oleoducto RAG | RAG 管线
BLOQUE5
### Uso de CLI | Uso en línea de comandos
BLOQUE6
### API Server | API 服务
BLOQUE7
Luego llama a la API:
BLOQUE8
## Configuración | 配置
Establecer variables de entorno:
BLOQUE9
## Desarrollo | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
O con docker-compose:









BLOCK12
## Estructura del proyecto | 项目结构
BLOCK13
## Licencia | Licencia
Licencia MIT — véase archivo [LICENCIA](LICENCIA).
## Contribuyendo | 贡献
¡Se agradecen las contribuciones! Por favor, consulte [CONTRIBUTING.md](CONTRIBUTING.md) para las directrices.
¡Se agradecen las contribuciones! Por favor, consulte [CONTRIBUTING.md](CONTRIBUTING.md) para las directrices de contribución.