# AI Agent with Tool Calling (ReAct-style)

A minimal but complete implementation of the agent pattern behind most modern "AI Agents" products: an LLM reasons about a goal, decides when it needs a tool, calls it, reads the result, and loops until it can answer.

## How it works

1. The user sends a goal in natural language (e.g. *"What is RAG, and how many chunks is 1200 documents at 180 tokens each?"*)
2. The LLM (via the Anthropic Messages API) is given a list of available tools as JSON Schema and decides whether it needs one
3. If it requests a tool call, Python executes the corresponding function and the result is appended to the conversation as a `tool_result`
4. The loop repeats (up to `MAX_TURNS`) until the model responds with plain text instead of a tool call, at which point that becomes the final answer

## Tools implemented

| Tool | Purpose |
|---|---|
| `calculator` | Safely evaluates basic arithmetic expressions |
| `current_time` | Returns the current UTC timestamp |
| `search_knowledge_base` | Looks up short definitions (LoRA, RAG, MCP) from a small local dictionary, a stand-in for a real retrieval/RAG tool |

## Why this matters

This is the same orchestration pattern used in production agent frameworks (LangChain agents, OpenAI function calling, Anthropic tool use): the LLM never executes code itself, it only decides *what* to call and *when to stop*. Understanding and being able to build this loop from scratch, without a heavy framework, is a strong signal for GenAI/Agent engineering roles.

## Requirements

```
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```
python agent.py
```

## Extending it

Swap `search_knowledge_base` for a real vector search (see the [`rag-pipeline-chroma`](../rag-pipeline-chroma) project in this repo), add a web-search or SQL tool, or point `TOOL_FUNCTIONS` at real internal APIs, the agent loop itself does not change.
