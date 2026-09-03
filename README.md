# MarketPilot

### Agentic AI-Powered Financial Research Assistant

MarketPilot is an AI-powered financial research assistant designed to help users analyze financial documents and answer finance-related questions using Retrieval-Augmented Generation (RAG).

The system combines document retrieval, LLM-based reasoning, and an agent workflow to provide answers grounded in the available financial documents.

---

## 🚀 Features

- **Agentic Workflow** — Uses LangGraph to coordinate planning, retrieval, analysis, and response generation.
- **Hybrid RAG** — Combines semantic vector search with BM25 keyword search for better document retrieval.
- **Financial Q&A** — Ask questions about financial reports and company information.
- **Company Comparison** — Compare financial metrics between companies.
- **Source Citations** — Responses include the documents used to generate the answer.
- **Local LLM Support** — Runs with Ollama for local and privacy-friendly inference.
- **OpenAI Support** — Can also be configured to use OpenAI models.
- **REST API** — FastAPI backend for integrating MarketPilot with other applications.
- **Streaming Responses** — Supports Server-Sent Events (SSE) for streaming generated responses.
- **Interactive API Documentation** — Swagger UI is available through FastAPI.

---

## 🧠 How It Works

MarketPilot follows an agent-based workflow to process financial questions.

```text
                    User Query
                        │
                        ▼
                 ┌─────────────┐
                 │    Planner  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Retrieval  │
                 └──────┬──────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Semantic Search          BM25 Search
             │                     │
             └──────────┬──────────┘
                        ▼
                 Hybrid Retrieval
                        │
                        ▼
                 ┌─────────────┐
                 │   Analysis  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Final Answer│
                 └─────────────┘

The retrieval layer combines semantic and keyword-based results before passing relevant context to the language model.

🛠️ Tech Stack
Technology	Purpose
Python	Core application
LangGraph	Agent workflow orchestration
LlamaIndex	Document indexing and retrieval
ChromaDB	Vector database
BM25	Keyword-based retrieval
Ollama	Local LLM inference
FastAPI	REST API
Pydantic	API data validation
pytest	Testing
📂 Project Structure
MarketPilot/
│
├── src/
│   ├── agents/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── ollama_client.py
│   │   └── openai_client.py
│   │
│   └── rag/
│       ├── indexer.py
│       ├── retriever.py
│       └── evaluator.py
│
├── examples/
├── scripts/
├── tests/
├── config.yaml
├── requirements.txt
├── pyproject.toml
└── README.md