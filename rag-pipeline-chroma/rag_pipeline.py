"""
Minimal Retrieval-Augmented Generation (RAG) pipeline.

Ingests a set of text documents, chunks them, embeds the chunks with a
local sentence-transformers model, stores the embeddings in a local
ChromaDB vector database, and at query time retrieves the most relevant
chunks to ground an LLM's answer, the standard pattern for giving an
LLM accurate, up-to-date, domain-specific knowledge without fine-tuning.
"""

import os
import textwrap

import chromadb
from chromadb.utils import embedding_functions
import anthropic

MODEL = os.environ.get("LLM_MODEL", "set-your-model-id-here")
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks, preserves context across cuts
TOP_K = 3              # number of chunks retrieved per query

DOCUMENTS = {
    "lora.txt": (
        "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. "
        "Instead of updating every weight in a large pretrained model, LoRA freezes "
        "the original weights and injects small trainable low-rank matrices into "
        "each layer. This drastically reduces the number of trainable parameters "
        "(often by 10000x) and the GPU memory needed for fine-tuning, while "
        "preserving most of the quality of full fine-tuning. QLoRA extends this by "
        "also quantizing the frozen base model to 4-bit precision, enabling "
        "fine-tuning of multi-billion parameter models on a single consumer GPU."
    ),
    "rag.txt": (
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
        "generative language model. At query time, relevant documents or chunks "
        "are retrieved from an external knowledge source, typically via vector "
        "similarity search, and inserted into the model's context window before "
        "it generates an answer. This grounds the model's output in real, "
        "up-to-date, and domain-specific information, reducing hallucination and "
        "avoiding the need to fine-tune the model every time the knowledge base "
        "changes."
    ),
    "mcp.txt": (
        "The Model Context Protocol (MCP) is an open standard that lets AI "
        "applications connect to external tools, data sources, and systems in a "
        "consistent way. An MCP server exposes a set of tools, resources, and "
        "prompts; any MCP-compatible client (such as an AI agent or assistant) "
        "can discover and call them without custom integration code per tool. "
        "This plays a similar role for AI agents that USB-C plays for hardware: "
        "one standard connector instead of one-off integrations."
    ),
    "agents.txt": (
        "An AI agent is a system where a language model decides, in a loop, which "
        "actions to take, such as calling a tool, querying a database, or asking "
        "a clarifying question, based on a goal and the results of its previous "
        "actions, rather than following a fixed script. Multi-agent systems "
        "coordinate several specialized agents (e.g. a planner, a researcher, and "
        "a writer) that communicate to solve a task none of them could solve "
        "alone."
    ),
}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def build_vector_store():
    """Ingest DOCUMENTS into a local, in-memory ChromaDB collection."""
    client = chromadb.Client()
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(name="ai_glossary", embedding_function=embedder)

    ids, texts, metadatas = [], [], []
    for doc_name, content in DOCUMENTS.items():
        for i, chunk in enumerate(chunk_text(content)):
            ids.append(f"{doc_name}-{i}")
            texts.append(chunk)
            metadatas.append({"source": doc_name})

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return collection


def retrieve(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def answer_with_rag(collection, query: str) -> str:
    """Retrieve relevant context, then ask the LLM to answer grounded in it."""
    chunks = retrieve(collection, query)
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)

    prompt = textwrap.dedent(f"""
        Answer the question using ONLY the context below. If the context does
        not contain the answer, say so instead of guessing.

        Context:
        {context}

        Question: {query}
    """).strip()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    store = build_vector_store()
    question = "How does QLoRA reduce GPU memory requirements during fine-tuning?"
    print(f"Q: {question}\n")
    print(f"A: {answer_with_rag(store, question)}")
