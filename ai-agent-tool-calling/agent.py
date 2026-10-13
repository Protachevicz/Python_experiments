"""
ReAct-style AI agent with tool calling (Anthropic Messages API).

The agent receives a natural-language goal, decides which tool to call
(if any), executes it, feeds the result back to the model, and repeats
until it can answer directly. This is the core loop behind most modern
"AI agents": the LLM does the reasoning/orchestration, Python code does
the actual work (math, lookups, APIs, file access, etc.).
"""

import json
import os
from datetime import datetime, timezone

import anthropic

MODEL = os.environ.get("LLM_MODEL", "set-your-model-id-here")
MAX_TURNS = 6


# ---------------------------------------------------------------------------
# Tools the agent is allowed to call. Each tool is a plain Python function
# plus a JSON Schema description the model uses to decide when/how to call it.
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def current_time(timezone_name: str = "UTC") -> str:
    """Return the current date/time (UTC only in this demo)."""
    return datetime.now(timezone.utc).isoformat()


KNOWLEDGE_BASE = {
    "lora": "LoRA (Low-Rank Adaptation) fine-tunes large models by training small "
            "low-rank matrices instead of all weights, cutting memory/compute cost.",
    "rag": "RAG (Retrieval-Augmented Generation) retrieves relevant documents and "
           "injects them into the prompt so the LLM answers using up-to-date, "
           "grounded context instead of relying only on its training data.",
    "mcp": "MCP (Model Context Protocol) is a standard for connecting LLM "
           "applications to external tools, data sources, and systems.",
}


def search_knowledge_base(query: str) -> str:
    """Look up a short definition from a small local knowledge base."""
    key = query.strip().lower()
    for term, definition in KNOWLEDGE_BASE.items():
        if term in key:
            return definition
    return "No matching entry found in the knowledge base."


TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression (+, -, *, /, parentheses).",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "current_time",
        "description": "Get the current UTC date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_knowledge_base",
        "description": "Search a small internal knowledge base of AI/ML terms "
                        "(e.g. LoRA, RAG, MCP) for a short definition.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": lambda **kw: calculator(kw["expression"]),
    "current_time": lambda **kw: current_time(kw.get("timezone_name", "UTC")),
    "search_knowledge_base": lambda **kw: search_knowledge_base(kw["query"]),
}


def run_agent(goal: str) -> str:
    """Run the agent loop until it produces a final text answer or hits MAX_TURNS."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": goal}]

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            # Model answered directly; return the final text.
            return "".join(block.text for block in response.content if block.type == "text")

        tool_results = []
        for tool_use in tool_uses:
            print(f"[turn {turn}] calling tool: {tool_use.name}({tool_use.input})")
            result = TOOL_FUNCTIONS[tool_use.name](**tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return "Agent stopped: reached the maximum number of turns without a final answer."


if __name__ == "__main__":
    question = (
        "What is RAG, and if I have 1200 documents split into chunks of 180 "
        "tokens each, roughly how many chunks is that? Also tell me the current UTC time."
    )
    print(json.dumps({"goal": question}, indent=2))
    print("\n--- Agent run ---\n")
    print(run_agent(question))
