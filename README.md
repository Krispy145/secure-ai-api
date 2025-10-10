# Secure AI API

FastAPI service for RAG + inference with OAuth2/JWT, rate limiting, Docker, CI/CD.

---

## 📈 Status

- **Status:** scaffolded (initial setup complete)
- **Focus:** Production-ready AI API with security best practices
- **Last updated:** 07/10/2025
- **Upcoming integration:** Phishing classifier and RAG endpoints

---

## 🔑 Highlights

- **AI Endpoints:** Phishing detection and RAG (Retrieval-Augmented Generation)
- **Authentication:** OAuth2/JWT with secure token handling
- **Security:** Rate limiting, input validation, and CORS protection
- **Infrastructure:** Docker containerization and CI/CD pipelines
- **Monitoring:** Health checks, logging, and performance metrics
- **Documentation:** Auto-generated OpenAPI/Swagger docs
- **Testing:** Comprehensive test suite with coverage reporting
- **Deployment:** Production-ready with environment configuration

---

## 🏗 Architecture Overview

Clean FastAPI architecture with security layers:

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
- Comprehensive testing and monitoring strategies

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

## 🔒 Security Features

- **OAuth2/JWT Authentication** with secure token validation
- **Rate Limiting** to prevent abuse and ensure fair usage
- **Input Validation** with Pydantic models and custom validators
- **CORS Protection** with configurable allowed origins
- **Environment-based Configuration** for secure secret management
- **Request Logging** for security monitoring and auditing

---

## 🗓 Roadmap

| Milestone                    | Category          | Target Date | Status     |
| ---------------------------- | ----------------- | ----------- | ---------- |
| Scaffold repo                | Backend Development | 12/10/2025  | ✅ Done    |
| Stub endpoints               | Backend Development | 15/10/2025  | ⏳ Pending |
| Phishing classifier integration | Backend Development | 20/10/2025 | ⏳ Planned |
| RAG endpoint implementation  | Backend Development | 24/10/2025  | ⏳ Planned |
| Docker + CI/CD setup         | Backend Development | 28/10/2025  | ⏳ Planned |
| JWT auth + rate limiting     | Backend Development | 04/11/2025  | ⏳ Planned |

---

## 📄 License

MIT © Krispy145