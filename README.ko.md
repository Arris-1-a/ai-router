<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# 아이-라우터
<p align="중심">
  <b>스마트 LLM API 라우터</b><br>
  지능형 라우팅 · 부하 분산 · 후퇴 · 비용 최적화<br>
  스마트 라우팅 · 부하 분산 · 장애 조치 · 비용 최적화
</p>
## 개요 | 개요
**ai-router**는 지능형 LLM API 관리를 위한 운영 준비가 된 Python 툴킷입니다. OpenAI, Anthropic, DeepSeek, Google 등 여러 AI 제공업체 간 요청을 라우팅할 수 있는 통합 인터페이스를 제공하며, 자동 부하 분산, 대체 조치, 비용 최적화, 포괄적인 지표를 지원합니다.
**ai-router**는 LLM API 요청을 지능적으로 관리할 수 있는 프로덕션 등급의 Python 툴킷입니다. OpenAI, Anthropic, DeepSeek, Google 등 여러 AI 제공업체 간 지능형 라우팅을 지원하는 통합 인터페이스를 제공하며, 자동 부하 분산, 장애 조치, 비용 최적화, 포괄적인 지표 모니터링 기능을 제공합니다.
## 건축 | 架构
블록0
## 특징 | 功能特性
### 🧠 스마트 라우터 | 智能路由器
- **다중 제공자**: OpenAI, Anthropic, DeepSeek, Google Gemini, 확장 가능
- **전략**: 라운드로빈, 가중치, 최소 지연, 최저 비용, 의미론적, 적응형
- **캐싱**: LRU + 의미 유사성 삭제 캐시
- **대체 차단기**: 회로 차단기, 자동 재시도, 다단계 대체 체인
- **속도 제한**: 토큰 버킷 알고리즘, 제공자별 및 전역 제한
- **지표**: 지연 시간, 비용, 성공률 추적과 경고
### 📚 RAG 파이프라인 | RAG 관선








- **청킹**: 고정 크기, 문장, 단락, 재귀, 마크다운, 슬라이딩 윈도우
- **임베딩**: OpenAI, 문장 변환기, 확장 가능한 백엔드
- **회수**: 하이브리드 (BM25 + 벡터), RRF 융합, 가중 결합
- **재순위 조정**: 점수 기반, MMR 다양성, 교차 인코더, LLM 심사
### 🤖 에이전트 프레임워크 | 에이전트 프레임
- **ReAct 에이전트**: 추론 + 도구 사용과 연기 루프
- **도구 시스템**: 데코레이터 기반 등록, JSON 스키마, 검증
- **오케스트레이션**: 순차적, 병렬적, 토론, 관리자-근로자 패턴
- **기억**: 단기, 장기, 에피소드, 작업 기억
### 📊 평가 | 评估
- **점수 산정**: BLEU, ROUGE-1/2/L, 의미 유사성, F1, 정확 일치
- **벤치마킹**: 지연 시간, 처리량, 비용, 부하 하 성공률
### 🌐 API 서버 | API 서비스
- **FastAPI**: OpenAI 호환 INLINE14, 임베딩, RAG, 에이전트 엔드포인트
- **미들웨어**: 로깅, 속도 제한, 요청 ID, 타이밍, CORS
## 퀵 스타트 | 快速开始
### 설치 | 安装
블록 1
또는 출처에서:
블록2




















### 기본 사용 | 기본 사용법
블록3
### 전략으로 루딩 | 策略路由
블록4
### RAG 파이프라인 | RAG 관선
블록5
### CLI 사용 | 명령줄 사용
블록6
### API 서버 | API 서비스
블록7
그 다음 API를 호출합니다:
블록8
## 구성 | 配置
환경 변수 설정:
블록9
## 개발 | 开发
BLOCK10
## 도커 | Docker 部署
BLOCK11
또는 docker-compose를 사용하면:









BLOCK12
## 프로젝트 구조 | 项目结构
BLOCK13
## 면허증 | 라이선스
MIT 라이선스 — [LICENSE](LICENSE) 파일을 참조하세요.
## 기여 | 贡献
기여 환영합니다! 지침은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하시기 바랍니다.
기여 환영합니다! 기여 지침은 [CONTRIBUTING.md](CONTRIBUTING.md)을 참조해 주시기 바랍니다.