<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=500&color=10B981&center=true&vCenter=true&width=700&height=70&lines=Food+Recipe+AI+Microservice;Groq+Llama+3.1+%E2%80%A2+Gemini+2.0+Flash" alt="Typing SVG" />
  </a>
  <p align="center">
    <strong>Python FastAPI AI Service for Recipe Extraction & Prompt Engineering</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-v3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/FastAPI-v0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
    <img src="https://img.shields.io/badge/Primary_LLM-Groq_Llama_3.1_8B-orange?style=for-the-badge" alt="Groq Llama 3.1"/>
    <img src="https://img.shields.io/badge/Fallback_LLM-Gemini_2.0_Flash-purple?style=for-the-badge&logo=google" alt="Gemini Flash"/>
    <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  </p>
</div>

# Food Recipe AI Microservice

Dịch vụ AI Microservice (Python FastAPI) đóng vai trò là động cơ xử lý trí tuệ nhân tạo (Intelligence Engine) độc lập cho hệ sinh thái **Món Gì Hôm Nay**.

Microservice nhận dữ liệu thô (tiêu đề, mô tả, lời thoại video từ YouTube), sau đó áp dụng kỹ thuật Prompt Engineering nâng cao trên các mô hình LLM (Groq Llama 3.1 8B & Google Gemini 2.0 Flash) để bóc tách, làm sạch dữ liệu và trả về công thức nấu ăn chuẩn cấu trúc JSON.

---

## 🏗️ Kiến trúc Fallback LLM 3 Tầng

```text
Backend Server (Node.js Express)
       |
       |-- POST /extract/batch
       v
FastAPI Router (/routers/extract.py)
       |
       |-- Context Builder (build_recipe_context)
       v
3-Tier LLM Engine (call_llm)
       |
       |---> [Tier 1] Groq Llama 3.1 8B Instant (Tốc độ cao, JSON Mode)
       |---> [Tier 2] Groq Llama 3.3 70B Versatile (Fallback khi Llama 8B gặp Rate Limit)
       |---> [Tier 3] Google Gemini 2.0 Flash (Fallback khi Groq hết Quota)
       v
Data Cleaning & Localizer (Quy đổi đơn vị đo lường Việt Nam)
       |
       v
Standardized Recipe JSON Response
```

---

## ⚡ Điểm Nổi bật về Kỹ thuật

1. **Kiến trúc Dự phòng LLM 3 Tầng (3-Tier LLM Fallback):** 
   - Đảm bảo tính sẵn sàng cao (High Availability) khi khai thác các API Free Tier.
   - Tự động chuyển đổi mượt mà giữa **Groq Llama 3.1 8B**, **Groq Llama 3.3 70B** và **Google Gemini 2.0 Flash** khi có sự cố nghẽn mạng hoặc hết hạn ngạch.

2. **Tối ưu Chi phí & Token (Batch Extraction):**
   - Hỗ trợ endpoint trích xuất hàng loạt (`POST /extract/batch`), cho phép gom nhiều video YouTube và trích xuất dữ liệu chỉ trong **1 lượt gọi LLM**, tiết kiệm tới 75% chi phí Token.

3. **Chuẩn hóa & Việt hóa Dữ liệu:**
   - Tự động làm sạch tiêu đề rác và link quảng cáo.
   - Quy đổi tự động các đơn vị quốc tế (`Tbsp`, `Tsp`, `cup`, `clove`) sang đơn vị Việt Nam (`muỗng canh`, `muỗng cà phê`, `chén`, `tép`).
   - Sửa lỗi cú pháp phân số JSON (`1/2` ➔ `0.5`, `1/3` ➔ `0.33`).

---

## 🔗 REST API Endpoint Chính

### `POST /extract/batch`

Trích xuất công thức từ danh sách video YouTube trong 1 request duy nhất.

#### Payload Mẫu:
```json
{
  "videos": [
    {
      "id": "yt_video_id_01",
      "title": "Cách làm Cá lóc kho tộ chuẩn vị",
      "description": "Mô tả video gốc...",
      "transcript": "Lời thoại phụ đề video..."
    }
  ]
}
```

#### Kết quả Phản hồi (JSON Response):
```json
{
  "recipes": [
    {
      "title": "Cá Lóc Kho Tộ",
      "description": "Món cá lóc kho tộ chuẩn vị miền Tây với thịt cá săn chắc, nước kho sóng sánh đậm đà đưa cơm.",
      "ingredients": [
        { "name": "Cá lóc", "amount": 500, "unit": "g", "category": "thịt" },
        { "name": "Nước mắm", "amount": 2, "unit": "muỗng canh", "category": "gia vị" }
      ],
      "steps": [
        {
          "stepNumber": 1,
          "phase": "prep",
          "instruction": "Sơ chế cá lóc: Rửa cá với nước muối pha loãng..."
        }
      ],
      "tags": ["cá lóc", "kho tộ"]
    }
  ]
}
```

### `GET /health`
Kiểm tra trạng thái hoạt động của microservice: `{"status": "ok", "service": "food-recipe-ai-service"}`.

---

## ⚙️ Cấu hình Biến môi trường (.env)

Tạo file `.env` trong thư mục gốc:

```env
PORT=8000
GROQ_API_KEY="your_groq_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
GROQ_MODEL="llama-3.1-8b-instant"
GEMINI_MODEL="gemini-2.0-flash"
```

---

## 🐳 Triển khai Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Lệnh build và chạy container:

```bash
docker build -t food-recipe-ai-service .
docker run -d -p 8000:8000 --env-file .env food-recipe-ai-service
```

---

<div align="center">
  <p>
    <strong>Food Recipe AI Microservice</strong> &copy; 2026 — Developed by <a href="https://github.com/phanttuan">Phan Viet Tuan</a>
  </p>
  <p>
    <em>Python FastAPI Microservice for Batch LLM Recipe Extraction.</em>
  </p>
</div>
