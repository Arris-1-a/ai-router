<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# AI-routeur
<p align="center">
  <b>Routeur</b> <br>API LLM intelligent
  Routage intelligent · Équilibrage de charge · Solution de secours · Optimisation <br>des coûts
  Routage intelligent · Équilibrage de charge · Basculement · Optimisation des coûts
</p>
## Aperçu | Aperçu
**ai-router** est une boîte à outils Python prête à la production pour la gestion intelligente des API LLM. Il offre une interface unifiée pour le routage des requêtes entre plusieurs fournisseurs d’IA (OpenAI, Anthropic, DeepSeek, Google, et d’autres) avec un équilibrage automatique de charge, un plan de secours, l’optimisation des coûts et des indicateurs complets.
**ai-router** est une boîte à outils Python de qualité production pour gérer intelligemment les requêtes API LLM. Il offre une interface unifiée supportant un routage intelligent entre plusieurs fournisseurs d’IA (OpenAI, Anthropic, DeepSeek, Google, etc.), comprenant un équilibrage automatique de charge, un basculement, l’optimisation des coûts et une surveillance complète des métriques.
## Architecture | 架构
BLOCK0
## Fonctionnalités | 功能特性
### 🧠 Routeur intelligent | 智能路由器
- **Multi-Fournisseur** : OpenAI, Anthropic, DeepSeek, Google Gemini, extensible
- **Stratégies** : Round-robin, pondéré, latence la plus faible, coût le plus faible, sémantique, adaptatif
- **Mise en cache** : LRU + cache dédoté par similarité sémantique
- **Solution de secours** : Disjoncteur, réessayage automatique, chaînes de secours multi-niveaux
- **Limitation de débit** : algorithme de bucket de jetons, par fournisseur et limites globales
- **Métriques** : Suivi de la latence, du coût et du taux de réussite avec les alertes
### 📚 Pipeline RAG | RAG 管线








- **Fragmentation** : taille fixe, phrase, paragraphe, récursive, markdown, fenêtre coulissante
- **Embedding** : OpenAI, transformateurs de phrases, backends extensibles
- **Récupération** : Hybride (BM25 + vecteur), fusion RRF, combinaison pondérée
- **Reclassement** : Basé sur les scores, diversité MMR, encodeur croisé, juge LLM
### 🤖 Cadre d’agent | Agent 框架
- **ReAct Agent** : Boucle de raisonnement + action avec l’utilisation de l’outil
- **Tool System** : Enregistrement basé sur le décorateur, schéma JSON, validation
- **Orchestration** : Schémas séquentiels, parallèles, débat, manager-employé
- **Mémoire** : Mémoire de travail à court terme, à long terme, épisodique
### 📊 Évaluation | 评估
- **Score** : BLEU, ROUGE-1/2/L, similarité sémantique, F1, correspondance exacte
- **Benchmarking** : Latence, débit, coût, taux de réussite sous charge
### 🌐 API Server | API 服务
- **FastAPI** : INLINE14 compatibles OpenAI, embeddings, RAG, terminaux d’agent
- **Middleware** : Journalisation, limitation de fréquence, ID de requête, synchronisation, CORS
## Démarrage rapide | 快速开始
### Installation | 安装
BLOC1
Ou d’après la source :
BLOCK2




















### Usage de base | Usage de base
BLOCK3
### Itinéraire avec stratégie | 策略路由
BLOCK4
### Pipeline RAG | RAG 管线
BLOCK5
### Utilisation du CLI | Utilisation en ligne de commande
BLOCK6
### API Server | API 服务
BLOCK7
Ensuite, appelez l’API :
BLOCK8
## Configuration | 配置
Définir les variables d’environnement :
BLOCK9
## Développement | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Ou avec docker-compose :









BLOCK12
## Structure du projet | 项目结构
BLOCK13
## Licence | Licence
Licence MIT — voir le dossier [LICENCE](LICENCE).
## Contribution | 贡献
Vos contributions sont les bienvenues ! Veuillez consulter [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.
Vos contributions sont les bienvenues ! Veuillez consulter [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives de contribution.