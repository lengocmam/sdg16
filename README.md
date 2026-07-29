# SDG16 Intelligence Platform

Nền tảng phân tích và dự đoán `Goal16` (chỉ số SDG 16 – Hòa bình, Công lý và
Thể chế vững mạnh) cho Việt Nam, triển khai đường chạy xuyên suốt:

`SDSN data → Data Lake → PySpark → Spark MLlib → Explainability → RAG → LLM → FastAPI → Spring Boot → Vue`

Dự án hiện thực hóa framework học máy có khả năng diễn giải (explainable ML)
được trình bày trong đồ án nghiên cứu *"An Explainable Machine Learning
Framework for SDG 16 Intelligence: Identifying Vietnam's Governance Gaps,
Forecasting Toward 2030, and Mapping Subnational Policy Priorities"*, nhằm
chẩn đoán khoảng trống quản trị của Việt Nam, dự báo quỹ đạo điểm số đến 2030,
và ánh xạ ưu tiên chính sách cấp tỉnh. Kiến trúc được thiết kế để chia việc độc
lập theo lớp (data/ML, backend, frontend) trong nhóm.

> Kết quả của hệ thống nên được hiểu là công cụ hỗ trợ ra quyết định
> (decision-support), không phải bằng chứng đánh giá tác động chính sách
> mang tính nhân quả.

## Mục lục

- [0. Bối cảnh nghiên cứu](#0-bối-cảnh-nghiên-cứu)
- [1. Kiến trúc](#1-kiến-trúc)
- [2. Chạy nhanh](#2-chạy-nhanh)
- [3. Chuẩn bị dữ liệu và train baseline](#3-chuẩn-bị-dữ-liệu-và-train-baseline)
- [4. Bật RAG và LLM](#4-bật-rag-và-llm)
- [5. Các profile Docker](#5-các-profile-docker)
- [6. Các quyết định dữ liệu quan trọng](#6-các-quyết-định-dữ-liệu-quan-trọng)
- [7. Lộ trình triển khai](#7-lộ-trình-triển-khai)
- [8. Biến môi trường](#8-biến-môi-trường)
- [9. Trạng thái các thành phần](#9-trạng-thái-các-thành-phần)
- [10. Đóng góp](#10-đóng-góp)
- [Tham khảo](#tham-khảo)
- [Giấy phép](#giấy-phép)

## 0. Bối cảnh nghiên cứu

SDSN công bố điểm SDG16 tổng hợp cho từng quốc gia (Sustainable Development
Report – SDR2024) nhưng không công khai đầy đủ cơ chế tổng hợp, khiến nhà
hoạch định chính sách khó xác định chỉ số thành phần nào đang kéo điểm số
xuống. Đồ án gốc giải quyết vấn đề này bằng pipeline phân tích 6 giai đoạn;
nền tảng này là bản triển khai hóa (production-oriented) của pipeline đó.

**Dữ liệu nghiên cứu (SDR2024 panel):**
- 4.392 quan sát quốc gia-năm, 183 quốc gia, giai đoạn 2000–2023
- 17 chỉ số SDG16 đầu vào đã chuẩn hóa (`n_sdg16_*`), biến mục tiêu `goal16`
- Chia theo thời gian để tránh leakage: Train ≤2018, Validation 2019–2021, Test ≥2022

**Dữ liệu cấp tỉnh (PAPI/PCI, dùng cho lớp explainability/RAG mở rộng):**
- PAPI: ~15.000 người dân được khảo sát/năm, 63 tỉnh (CECODES & UNDP)
- PCI: ~10.000 doanh nghiệp được khảo sát/năm, 63 tỉnh (VCCI)
- 310 quan sát, giai đoạn 2019–2023

**Sáu giai đoạn phân tích của đồ án và ánh xạ sang các khối hệ thống:**

| Giai đoạn nghiên cứu | Vai trò | Khối tương ứng trong hệ thống |
|---|---|---|
| 1. Panel OLS (Fixed Effects) | Baseline kinh tế lượng, kiểm soát hiệu ứng cố định theo quốc gia/năm | tham chiếu offline, chưa đưa vào API |
| 2. XGBoost / Spark MLlib | Dự đoán `goal16` từ 17 chỉ số | **ML baseline (Spark MLlib)** |
| 3. SHAP-style explanation | Diễn giải đóng góp từng chỉ số | **Explainability (coefficient contribution)** |
| 4. Subnational drill-down | Ánh xạ khoảng trống quốc gia → chỉ số PAPI/PCI cấp tỉnh | định hướng mở rộng RAG knowledge base |
| 5. GRU forecasting | Dự báo đệ quy Goal16 2024–2030 | định hướng mở rộng, chưa có trong pipeline hiện tại |
| 6. RAG-LLM | Sinh khuyến nghị chính sách có căn cứ | **RAG (Qdrant) + LLM** |

Baseline hiện tại của hệ thống dùng **Spark MLlib Linear Regression** để dự
đoán `goal16` và reverse-engineer hệ số của các chỉ số `n_sdg16_*` — đây cũng
chính là mô hình baseline tuyến tính được báo cáo trong đồ án gốc, dùng để đối
chứng với các mô hình phi tuyến (XGBoost, GRU) sẽ được tích hợp ở các giai
đoạn tiếp theo.

### Kết quả tham chiếu từ đồ án nghiên cứu

| Mô hình | Split | RMSE | MAE | R² |
|---|---|---|---|---|
| XGBoost (SHAP Runner) | Test | 1.8249 | 1.4019 | 0.9859 |
| Panel OLS + Fixed Effects | In-sample | 2.1276 | 1.6099 | 0.6263 |
| GRU Sequence Forecaster | Test | 2.4411 | 1.9119 | 0.9747 |
| Spark Linear Regression (baseline) | Test | 12.2139 | 9.2173 | 0.3509 |

Bảng trên là kết quả benchmark trong đồ án, dùng làm mốc so sánh khi hệ thống
tích hợp thêm XGBoost/GRU thay cho baseline tuyến tính hiện tại.

**Khoảng trống quản trị lớn nhất của Việt Nam (SHAP, 2022):**
1. Tự do báo chí / trách nhiệm giải trình — *sơ bộ, dữ liệu impute, cần kiểm chứng*
2. Tiếp cận công lý — đã kiểm chứng (điểm 60.77)
3. Bảo vệ quyền tài sản — *sơ bộ, dữ liệu impute, cần kiểm chứng*
4. Lao động trẻ em — đã kiểm chứng (điểm 74.58)
5. Minh bạch hành chính — đã kiểm chứng (điểm 3.78)

**Dự báo GRU:** điểm Goal16 của Việt Nam gần như đi ngang, từ 63.65 (2024)
đến 63.64 (2030) theo kịch bản baseline đã kiểm chứng — dùng làm mốc theo dõi
trung hạn, không phải bằng chứng cải thiện thể chế tự động.

Chi tiết thiết kế hệ thống: [docs/architecture.md](docs/architecture.md).

## 1. Kiến trúc

| Khối | Công nghệ | Trách nhiệm |
|---|---|---|
| Data lake | thư mục volume / MinIO profile | lưu raw, clean và artifact |
| Processing | PySpark, Spark SQL, Parquet | hợp nhất, chuẩn hóa, lọc feature |
| ML baseline | Spark MLlib | imputation, scaling, Linear/Ridge/Lasso |
| Explainability | coefficient contribution | giải thích chiều và mức tác động |
| RAG | Qdrant + embedding adapter | truy xuất báo cáo/chính sách |
| LLM | OpenAI-compatible hoặc Ollama | diễn giải và sinh khuyến nghị |
| ML service | FastAPI | inference, explainability, RAG và LLM nội bộ |
| Web backend | Spring Boot | public API, validation, error handling và BFF |
| Frontend | Vue 3 + Nginx | dashboard web responsive và reverse proxy |
| Deployment | Docker Compose profiles | chạy từng lớp theo tài nguyên máy |

## 2. Chạy nhanh

Yêu cầu: Docker Desktop với Docker Compose v2.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Sau khi khởi động:

- Web Vue: <http://localhost:3000>
- Spring Boot API: <http://localhost:8080/api/v1/health>
- Spring Boot health: <http://localhost:8080/actuator/health>
- FastAPI ML docs: <http://localhost:8000/docs>
- Qdrant: <http://localhost:6333/dashboard>

API vẫn khởi động khi chưa có model, nhưng `/predict` và `/explain` sẽ trả
`503 Model not ready`. Đây là trạng thái có chủ đích để nhóm Data/ML và nhóm
Backend có thể làm song song.

### Chạy frontend/backend khi phát triển

Spring Boot:

```powershell
cd apps/backend
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-21.0.9.10-hotspot"
mvn spring-boot:run
```

Vue:

```powershell
cd apps/frontend
npm install
npm run dev
```

Vite chạy tại <http://localhost:5173> và proxy `/api` tới Spring Boot port
`8080`. Khi chạy Docker, Nginx phục vụ bản build tại port `3000`.

Public API do Spring Boot cung cấp:

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/v1/health` | trạng thái ML/RAG/LLM |
| GET | `/api/v1/model` | metadata và metric model |
| POST | `/api/v1/predictions` | dự đoán Goal16 |
| POST | `/api/v1/explanations` | dự đoán và contribution |
| POST | `/api/v1/knowledge/search` | tìm tài liệu RAG |
| POST | `/api/v1/assistant/questions` | tạo câu trả lời chính sách |

## 3. Chuẩn bị dữ liệu và train baseline

Đặt các file SDSN CSV vào `data/raw/`. Tối thiểu dữ liệu cần có:

- `country`
- `year`
- `goal16`
- các cột số có prefix `n_sdg16_`

Tên cột được chuẩn hóa thành chữ thường và dấu gạch dưới. Một số alias như
`Country Name`, `Goal 16`, `SDG16 Score` được tự động ánh xạ.

Khi nhiều nguồn có cùng `(country, year)`, pipeline ưu tiên filename chứa các
token trong `data.source_priority` của `configs/project.yaml`. Hãy đổi danh sách
này cho khớp tên file thật trước khi chạy.

Chạy toàn bộ pipeline bằng Spark local trong container:

```powershell
docker compose --profile jobs run --rm pipeline python -m pipelines.run_pipeline all
docker compose restart ml-api backend
```

Kết quả:

- `data/clean/sdg16.parquet`: dataset sạch;
- `artifacts/linear_regression/spark_pipeline`: Spark PipelineModel;
- `artifacts/linear_regression/metadata.json`: hệ số, scaler, metrics và version.

Muốn chạy Spark cluster:

```powershell
docker compose --profile spark up -d spark-master spark-worker
$env:SPARK_MASTER_URL="spark://spark-master:7077"
docker compose --profile jobs run --rm pipeline python -m pipelines.run_pipeline all
```

## 4. Bật RAG và LLM

Mặc định RAG/LLM bị tắt để stack lõi chạy được mà không cần API key hoặc GPU.
Đây là khối triển khai của **Giai đoạn 6 (RAG-LLM Policy Recommendation)** trong
đồ án nghiên cứu: mô hình ngôn ngữ chỉ tổng hợp khuyến nghị từ tài liệu đã
truy xuất và kết quả các mô hình trước đó, không tự tính điểm hay suy luận
thống kê.

### Scan PDF local thành dữ liệu RAG

Bộ PDF hiện nằm trong:

```text
rag/tailieuLLM-20260626T154614Z-3-001/tailieuLLM
```

Parser sẽ tự nhận diện `Nhóm 1`–`Nhóm 5` để gắn metadata cho RAG. Chạy:

```powershell
docker compose --profile jobs run --rm pdf-parser
```

Output:

- `data/knowledge/processed/chunks.jsonl`
- `data/knowledge/processed/parse_report.json`
- `data/knowledge/text/*.txt`

Chi tiết: [docs/knowledge-ingestion.md](docs/knowledge-ingestion.md).

### OpenAI-compatible

Điền `.env`:

```dotenv
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=<embedding-model>
LLM_PROVIDER=openai-compatible
LLM_MODEL=<chat-model>
LLM_API_KEY=<secret>
LLM_BASE_URL=<provider-base-url>
```

Sau khi đã parse PDF hoặc đưa văn bản `.txt` vào `data/knowledge/text`, chạy:

```powershell
docker compose --profile jobs run --rm rag-indexer
docker compose restart ml-api backend
```

### Ollama local

```powershell
docker compose --profile local-llm up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
```

Sau đó đặt `LLM_PROVIDER=ollama`. Để embedding qua Ollama, đặt thêm
`EMBEDDING_PROVIDER=ollama` và chọn một embedding model đã pull.

Với demo hiện tại dùng Ollama `gemma3:4b` trên máy host và chưa cần embedding model,
đọc hướng dẫn nhanh tại [docs/rag-ollama-demo.md](docs/rag-ollama-demo.md).

## 5. Các profile Docker

| Lệnh | Thành phần |
|---|---|
| `docker compose up` | Vue + Spring Boot + FastAPI ML + Qdrant |
| `--profile jobs` | Spark pipeline, PDF parser và RAG indexer chạy theo job |
| `--profile spark` | Spark master/worker |
| `--profile data-lake` | MinIO |
| `--profile local-llm` | Ollama |

## 6. Các quyết định dữ liệu quan trọng

- Không thay missing value bằng `0` một cách mặc định. Với SDG, `0` có thể là
  một quan sát thật và việc lấp 0 dễ làm sai hệ số. Baseline dùng median học
  chỉ trên train split. *(Đồ án gốc dùng zero-imputation cho một số chỉ số
  như `n_sdg16_rsf`, `n_sdg16_exprop` và ghi nhận đây là hạn chế cần kiểm
  chứng — hệ thống này cố tình tránh lặp lại hạn chế đó.)*
- Chia train/validation/test theo thời gian, không random split, để tránh
  leakage từ tương lai.
- API chỉ dùng artifact đã version hóa; LLM không trực tiếp tính điểm.
- Hệ số linear là baseline giải thích được, chưa nên gọi là "trọng số chính
  thức" của SDSN nếu chưa kiểm định độ ổn định, fixed effects và robustness.
- SHAP/coefficient contribution thể hiện đóng góp trong mô hình, không phải
  quan hệ nhân quả — khuyến nghị chính sách sinh ra từ hệ thống cần kết hợp
  phân tích định tính trước khi áp dụng.

## 7. Lộ trình triển khai

Workflow chi tiết và tiêu chí hoàn thành nằm tại
[docs/workflow.md](docs/workflow.md). Ma trận phân công nằm tại
[docs/team-allocation.md](docs/team-allocation.md).

Định hướng mở rộng theo đồ án nghiên cứu:

1. Tích hợp XGBoost + SHAP TreeExplainer thay thế/bổ sung cho Spark Linear
   Regression baseline.
2. Thêm mô-đun GRU forecasting (Giai đoạn 5) để dự báo `goal16` 2024–2030.
3. Xây lớp subnational drill-down (Giai đoạn 4) ánh xạ SHAP gap sang chỉ số
   PAPI/PCI cấp tỉnh.
4. Kiểm chứng dữ liệu gốc cho các chỉ số hiện đang impute (`n_sdg16_rsf`,
   `n_sdg16_exprop`) trước khi dùng trong khuyến nghị chính sách.

## 8. Biến môi trường

Tối thiểu cần khai báo trong `.env` (copy từ `.env.example`):

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `SPARK_MASTER_URL` | Không (mặc định local) | Địa chỉ Spark master khi chạy cluster |
| `EMBEDDING_PROVIDER` | Khi bật RAG | `openai-compatible` hoặc `ollama` |
| `EMBEDDING_MODEL` | Khi bật RAG | Tên model embedding |
| `LLM_PROVIDER` | Khi bật RAG/LLM | `openai-compatible` hoặc `ollama` |
| `LLM_MODEL` | Khi bật RAG/LLM | Tên model chat |
| `LLM_API_KEY` | Khi dùng provider ngoài | API key của provider |
| `LLM_BASE_URL` | Khi dùng provider ngoài | Base URL của provider |
| `QDRANT_URL` | Không (mặc định `http://qdrant:6333`) | Địa chỉ Qdrant nội bộ trong Docker network |

*(Rà lại danh sách này theo `.env.example` thật của repo trước khi merge —
đây là tập hợp tối thiểu suy ra từ các bước cấu hình ở trên.)*

## 9. Trạng thái các thành phần

| Thành phần | Trạng thái |
|---|---|
| Data lake + PySpark pipeline | Sẵn sàng |
| Spark MLlib Linear Regression baseline | Sẵn sàng |
| FastAPI ML service (`/predict`, `/explain`) | Sẵn sàng, trả `503` khi chưa có model |
| Spring Boot public API | Sẵn sàng |
| Vue dashboard | Sẵn sàng |
| RAG (PDF parser + Qdrant indexer) | Sẵn sàng, tắt mặc định |
| LLM (OpenAI-compatible / Ollama) | Sẵn sàng, tắt mặc định |
| XGBoost + SHAP TreeExplainer | Chưa triển khai — kế hoạch |
| GRU forecasting (2024–2030) | Chưa triển khai — kế hoạch |
| Subnational drill-down (PAPI/PCI) | Chưa triển khai — kế hoạch |

## 10. Đóng góp

1. Tạo branch từ `main` theo quy ước `feature/<tên-việc>` hoặc `fix/<tên-việc>`.
2. Với thay đổi ở pipeline/model, chạy lại `run_pipeline all` và đính kèm
   `metadata.json` mới trong PR để nhóm review được thay đổi metric.
3. Với thay đổi API (Spring Boot/FastAPI), cập nhật bảng endpoint tương ứng
   trong README.
4. Mọi thay đổi liên quan tới dữ liệu impute hoặc cách chia split cần nêu rõ
   lý do trong PR, vì đây là các quyết định đã ảnh hưởng trực tiếp tới độ tin
   cậy kết quả nghiên cứu gốc.

## Tham khảo

- Sachs, J.D., Lafortune, G., & Fuller, G. (2024). *The SDGs and the UN
  Summit of the Future: Sustainable Development Report 2024*. SDSN.
- CECODES, RTA & UNDP (2024/2025). *The Viet Nam Provincial Governance and
  Public Administration Performance Index (PAPI)*.
- Malesky, E. (2023). *The Vietnam Provincial Competitiveness Index (PCI)*.
  VCCI.
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting
  model predictions*. NeurIPS.

## Giấy phép

*(Chưa xác định — thêm license phù hợp, ví dụ MIT hoặc Apache-2.0, trước khi
public repo.)*