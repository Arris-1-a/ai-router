<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# 人工智能路由器
<p align=“中心”>
<b>智能大型语言模型API路由器</b><br>
智能路由·负载均衡 ·后备方案 ·成本优化<br>
  智能路由 · 负载均衡 · 故障转移 · 成本优化
</p>
## Overview | 概述
**ai-router** 是一个面向生产环境的 Python 工具包，用于智能 LLM API 管理。它提供了一个统一的界面，用于跨多个 AI 提供商（OpenAI、Anthropic、DeepSeek、Google 等）路由请求，并具备自动负载均衡、备选、成本优化和全面的指标。
**ai-router** 是一个生产级的 Python 工具包，用于智能管理 LLM API 请求。 它提供统一接口，支持跨多个 AI 提供商（OpenAI、Anthropic、DeepSeek、Google 等）的智能路由，具备自动负载均衡、故障转移、成本优化和全面的指标监控。
## 建筑 |架构
BLOCK0
## 专题 |功能特性
### 🧠 智能路由器 |智能路由器
- **多提供者**：OpenAI、Anthropic、DeepSeek、Google Gemini、可扩展
- **策略**：轮询、加权、最低延迟、最低成本、语义、自适应
- **缓存**：LRU + 语义相似度去重缓存
- **备份**：断路器、自动重试、多级备份链
- **速率限制**：令牌桶算法，按提供者及全局限制
- **指标**：延迟、成本、成功率跟踪及警报
### 📚 RAG管道 |RAG 管线








- **分块**：固定大小、句子、段落、递归、标记、滑动窗口
- **嵌入**：OpenAI、句子变换器、可扩展后端
- **回收**：混合型（BM25+载体）、RRF聚变、加权组合
- **重新排名**：基于分数、MMR多样性、交叉编码器、LLM裁判
### 🤖 代理框架 |代理框架
- **反应代理**：推理 + 工具使用作用循环
- **工具系统**：基于装饰器的注册、JSON 模式、验证
- **编排：顺序、平行、辩论、管理者-员工模式
- **记忆**：短期、长期、情景记忆、工作记忆
### 📊 评估 |评估
- **评分**：BLEU、ROUGE-1/2/L、语义相似、F1、完全匹配
- **基准测试**：延迟、吞吐量、成本、负载下成功率
### 🌐 API 服务器 |API 服务
- **FastAPI**：兼容 OpenAI 的INLINE14、嵌入、RAG、代理端点
- **中间件**：日志记录、速率限制、请求ID、时序、CORS
## 快速启动 |快速开始
### 安装 |安装
第一区
或者引用来源：
第二区块




















### Basic Usage | 基本用法
第三区块
### 战略战术 |策略路由
第四区块
### RAG管道 |RAG 管线
第五区块
### CLI Usage | 命令行使用
第6区块
### API 服务器 |API 服务
第7区块
然后调用API：
第8区块
## 配置 |配置
设置环境变量：
第9区块
## 开发 |开发
BLOCK10
## Docker |Docker 部署
BLOCK11
或者用docker-compose：









BLOCK12
## 项目结构 |项目结构
BLOCK13
## License | 许可证
麻省理工学院许可证——参见[许可证]（许可证）文件。
## 贡献 |贡献
欢迎大家的贡献！请参阅[CONTRIBUTING.md]（CONTRIBUTING.md）获取指导方针。
欢迎贡献！ 请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。