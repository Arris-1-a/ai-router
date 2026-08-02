<div align="center">

**🌐 Language / 选择语言 / Idioma:**

[English](README.md) · [简体中文](README.zh-CN.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Português](README.pt.md) · [Русский](README.ru.md) · [اردو](README.ur.md) · [Bahasa Indonesia](README.id.md) · [Deutsch](README.de.md) · [日本語](README.ja.md) · [मराठी](README.mr.md) · [తెలుగు](README.te.md) · [Türkçe](README.tr.md) · [தமிழ்](README.ta.md) · [Tiếng Việt](README.vi.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---


---








# Bộ định tuyến AI
<p align="center">
  Bộ <b>định tuyến</b> <br>API Smart LLM
  Định tuyến thông minh · Cân bằng tải · Phương án dự phòng · Tối ưu hóa <br>chi phí
  Định tuyến thông minh · Cân bằng tải · Chuyển đổi dự phòng · Tối ưu hóa chi phí
</p>
## Tổng quan | Tổng quan
**ai-router** là bộ công cụ Python sẵn sàng cho sản xuất để quản lý API LLM thông minh. Nó cung cấp giao diện thống nhất để định tuyến yêu cầu giữa nhiều nhà cung cấp AI (OpenAI, Anthropic, DeepSeek, Google và nhiều hơn nữa) với cân bằng tải tự động, dự phòng, tối ưu hóa chi phí và các chỉ số toàn diện.
**ai-router** là bộ công cụ Python cấp sản xuất để quản lý thông minh các yêu cầu API LLM. Nó cung cấp giao diện thống nhất hỗ trợ định tuyến thông minh giữa nhiều nhà cung cấp AI (OpenAI, Anthropic, DeepSeek, Google, v.v.), bao gồm cân bằng tải tự động, chuyển đổi dự phòng, tối ưu hóa chi phí và giám sát chỉ số toàn diện.
## Kiến trúc | 架构
BLOCK0
## Tính năng | 功能特性
### 🧠 Bộ định tuyến thông minh | 智能路由器
- **Đa nhà cung cấp**: OpenAI, Anthropic, DeepSeek, Google Gemini, có thể mở rộng
- **Chiến lược**: Vòng tròn, có trọng số, độ trễ thấp nhất, chi phí thấp nhất, ngữ nghĩa, thích ứng
- **Bộ nhớ đệm**: LRU + bộ nhớ đệm dedup tương đồng ngữ nghĩa
- **Dự phòng**: Cầu dao mạch, tự động thử lại, chuỗi dự phòng đa cấp
- **Giới hạn tốc độ**: Thuật toán bucket token, giới hạn cho từng nhà cung cấp và toàn cầu
- **Chỉ số**: Theo dõi độ trễ, chi phí, tỷ lệ thành công kèm cảnh báo
### 📚 Đường ống RAG | RAG 管线








- **Chunking**: Câu, đoạn văn, đệ quy, đánh dấu, cửa sổ trượt có kích thước cố định
- **Nhúng**: OpenAI, bộ chuyển đổi câu, backend mở rộng
- **Thu hồi**: Hybrid (BM25 + vector), hợp nhất RRF, tổ hợp trọng số
- **Xếp hạng**: Dựa trên điểm số, đa dạng MMR, mã hóa chéo, giám khảo LLM
### 🤖 Khung Đại Lý | Agent 框架
- **ReAct Agent**: Suy luận + Vòng lặp hành động với việc sử dụng công cụ
- **Hệ thống công cụ**: Đăng ký dựa trên Decorator, Schema JSON, xác thực
- **Phối khí**: Tuần tự, song song, tranh luận, mô hình quản lý-công nhân
- **Trí nhớ**: Trí nhớ ngắn hạn, dài hạn, trí nhớ theo từng tập, trí nhớ làm việc
### 📊 Đánh giá | 评估
- **Điểm số**: BLEU, ROUGE-1/2/L, tương đồng ngữ nghĩa, F1, khớp chính xác
- **Đánh giá chuẩn**: Độ trễ, thông lượng, chi phí, tỷ lệ thành công khi tải
### 🌐 Máy chủ API | API 服务
- **FastAPI**: INLINE14 tương thích OpenAI, embeddings, RAG, các điểm cuối agent
- **Phần mềm trung gian**: Ghi nhật ký, giới hạn tốc độ, ID yêu cầu, thời gian, CORS
## Khởi Đầu Nhanh | 快速开始
### Cài đặt | 安装
BLOCK1
Hoặc từ nguồn:
BLOCK2




















### Cách sử dụng cơ bản | Cách sử dụng cơ bản
BLOCK3
### Định tuyến với Chiến lược | 策略路由
BLOCK4
### Đường ống RAG | RAG 管线
BLOCK5
### Cách sử dụng CLI | Cách sử dụng dòng lệnh
BLOCK6
### Máy chủ API | API 服务
BLOCK7
Sau đó gọi API:
BLOCK8
## Cấu hình | 配置
Thiết lập biến môi trường:
BLOCK9
## Phát triển | 开发
BLOCK10
## Docker | Docker 部署
BLOCK11
Hoặc với docker-compose:









BLOCK12
## Cấu trúc dự án | 项目结构
BLOCK13
## Giấy phép | Giấy phép
Giấy phép MIT — xem tệp [LICENSE](LICENSE).
## Đóng góp | 贡献
Mọi đóng góp đều được hoan nghênh! Vui lòng xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết hướng dẫn.
Mọi đóng góp đều được hoan nghênh! Vui lòng tham khảo [CONTRIBUTING.md](CONTRIBUTING.md) để biết hướng dẫn đóng góp.