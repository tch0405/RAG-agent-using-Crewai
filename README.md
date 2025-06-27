#  CrewAI RAG Agent with FAISS + Hugging Face + Serper

This project is a multi-agent system powered by [CrewAI](https://docs.crewai.com) that combines:

- 📚 RAG (Retrieval-Augmented Generation) using FAISS
- 💬 LLM via Hugging Face Inference API (e.g. Mistral-7B)
- 🛠 Custom CrewAI-compatible tools
- 🔐 Secure environment variable loading via `.env`

---

## 📦 Features

- Agent 1: **Research Analyst** — pulls current trends using Google Search
- Agent 2: **Knowledge Assistant** — answers questions from a local document knowledge base (FAISS)
- Uses Hugging Face `Mistral-7B-Instruct` via `langchain-huggingface`
- Simple, modular, and easily extendable with other tools

---

## 🚀 Quickstart

### 1. Clone this repo

```bash
git clone https://github.com/tch0405/crewai-rag-agent.git
cd crewai-rag-agent

```
### 2. Install dependencies
```bash
pip install -r requirements.txt

```
### 3. Set up .env
```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
OPENAI_API_KEY=your_openai_key_if_needed
```

### 4. the agent
```bash
python crewai_agent.py
```
