# RAG Pipeline with ChromaDB

A compact Retrieval-Augmented Generation (RAG) pipeline: chunk documents, embed them locally, store/search them in a vector database, and ground an LLM's answer in the retrieved context.

## Pipeline

1. **Ingest**: a handful of short AI/ML glossary documents (LoRA, RAG, MCP, Agents) are split into overlapping chunks (`chunk_text`) so relevant information isn't cut off at arbitrary boundaries
2. **Embed & store**: each chunk is embedded locally with a `sentence-transformers` model (`all-MiniLM-L6-v2`, no API cost) and stored in a local **ChromaDB** collection
3. **Retrieve**: at query time, the question is embedded and ChromaDB returns the top-k most semantically similar chunks (`retrieve`)
4. **Generate**: the retrieved chunks are inserted into the prompt as context, and the LLM is instructed to answer *only* from that context, reducing hallucination (`answer_with_rag`)

## Why this matters

RAG is the standard way to give an LLM accurate, current, and domain-specific knowledge without the cost and latency of fine-tuning. This same pattern, swap the toy glossary for real documents, scale ChromaDB to a managed vector store (Pinecone, pgvector, Weaviate), is what powers most production "chat with your docs" and enterprise knowledge-assistant products.

## Requirements

```
pip install chromadb sentence-transformers anthropic
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```
python rag_pipeline.py
```

## Extending it

- Swap the in-memory `chromadb.Client()` for `chromadb.PersistentClient(path=...)` to persist the index across runs
- Replace the toy `DOCUMENTS` dict with a loader over real files (PDF, Markdown, HTML)
- Feed `retrieve()`'s output into the [`ai-agent-tool-calling`](../ai-agent-tool-calling) project as a `search_knowledge_base` tool for a full agent + RAG system
