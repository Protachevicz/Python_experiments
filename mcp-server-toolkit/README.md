# MCP Server Toolkit (TypeScript)

A minimal **Model Context Protocol (MCP)** server, the open standard (introduced by Anthropic, now widely adopted across the AI ecosystem) for connecting AI applications to external tools and data sources in a consistent, discoverable way. Think of it as a universal connector: any MCP-compatible client (an AI assistant, an IDE, a custom agent) can discover and call these tools without one-off integration code per tool.

This server exposes three small research/data-science tools over the standard stdio transport.

## Tools exposed

| Tool | Purpose |
|---|---|
| `summary_statistics` | Computes mean, standard deviation, min, max, and median for a list of numbers |
| `ab_test_significance` | Runs a two-proportion z-test to check whether an A/B test result is statistically significant (ties into the [`experiment-design-ab-testing`](../experiment-design-ab-testing) folder in this repo) |
| `define_term` | Looks up short definitions of AI/ML terms (LoRA, RAG, MCP, Bayesian inference) from a small glossary |

## Why this matters

MCP is one of the most recent and fastest-adopted pieces of AI infrastructure (2024-2025): it decouples *tools* from *the specific agent/assistant that uses them*, the same server works unchanged whether it's called by a chat assistant, an IDE, or a custom multi-agent system. Building one from scratch (rather than only consuming existing MCP servers) demonstrates hands-on familiarity with current AI agent infrastructure, not just prompt-level agent usage.

## Requirements

- Node.js 18+

## Install & run

```bash
npm install
npm run dev        # run directly with tsx (no build step)
# or
npm run build && npm start
```

## Trying it with an MCP client

Point any MCP-compatible client (e.g. a local AI assistant with MCP support) at this server by adding it to the client's MCP config, for example:

```json
{
  "mcpServers": {
    "mcp-server-toolkit": {
      "command": "node",
      "args": ["/absolute/path/to/mcp-server-toolkit/dist/index.js"]
    }
  }
}
```

## Extending it

Add more `server.tool(...)` definitions following the same pattern, each tool is a name, a description, a `zod` input schema, and an async handler, no transport or protocol code required.
