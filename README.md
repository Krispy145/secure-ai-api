# Secure Ai Api

FastAPI service for RAG + inference with OAuth2/JWT, rate limiting, Docker, CI/CD.

---

## 📈 Status

- **Status:** active (Active)
- **Focus:** FastAPI service for RAG + inference with OAuth2/JWT, rate limiting, Docker, CI/CD.
- **Last updated:** 20/11/2025
- **Target completion:** 23/11/2025

---

## 🔑 Highlights

- **AI Endpoints** → Phishing detection and RAG (Retrieval-Augmented Generation)
- **Authentication** → OAuth2/JWT with secure token handling
- **Security** → Rate limiting, input validation, and CORS protection
- **Infrastructure** → Docker containerization and CI/CD pipelines
- **Monitoring** → Health checks, logging, and performance metrics
- **Documentation** → Auto-generated OpenAPI/Swagger docs

---

## 🏗 Architecture Overview

```
app/
 ├─ api/v1/         # router.py, phishing.py, rag.py
 ├─ core/           # config.py, security, middleware
 └─ main.py         # FastAPI application entry point
```

**Patterns used:**

- `api/v1/` contains versioned API endpoints
- `core/` handles configuration and security middleware
- `main.py` initializes the FastAPI application
- Docker configuration for containerized deployment
- GitHub Actions for automated CI/CD

---

## 📱 What It Demonstrates

- Production-ready FastAPI application structure
- Secure API design with authentication and authorization
- AI/ML model integration and inference endpoints
- Containerization and deployment best practices

---

## 🚀 Getting Started

```bash
git clone https://github.com/Krispy145/secure-ai-api.git
cd secure-ai-api
pip install -r requirements.txt
```

**Run locally:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Run with Docker:**
```bash
docker-compose up --build
```

**API Documentation:**
Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 🧪 Testing

```bash
pytest tests/ --cov=app --cov-report=html
```

- Unit tests → API endpoints and business logic
- Integration tests → Database and external service interactions
- Security tests → Authentication and authorization flows
- Performance tests → Load testing and rate limiting

---

## 🔒 Security & Next Steps

- Follow security best practices for the technology stack
- Implement proper authentication and authorization
- Add comprehensive error handling and validation
- Set up monitoring and logging

---

## 🗓 Roadmap

| Milestone                    | Category              | Target Date | Status     |
| ---------------------------- | --------------------- | ----------- | ---------- |
| Stub endpoints | Backend Development | 26/10/2025 | ✅ Done |
| Phishing classifier integration | Backend Development | 30/11/2025 | ✅ Done |
| RAG endpoint implementation | Backend Development | 30/11/2025 | ⏳ In Progress |
| Docker + CI/CD setup | Backend Development | 30/11/2025 | ⏳ In Progress |
| JWT auth + rate limiting | Backend Development | 06/12/2025 | ⏳ In Progress |


---

## 📄 License

MIT © Krispy145