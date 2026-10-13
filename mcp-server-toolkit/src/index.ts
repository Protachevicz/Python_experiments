#!/usr/bin/env node
/**
 * Minimal Model Context Protocol (MCP) server.
 *
 * MCP is an open standard for connecting AI applications to external tools,
 * data, and systems in a consistent, discoverable way, the same server
 * works with any MCP-compatible client without per-tool integration code.
 *
 * This server exposes three small research/data-science oriented tools over
 * stdio transport, the standard way to run a local MCP server.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "mcp-server-toolkit",
  version: "1.0.0",
});

// ---------------------------------------------------------------------------
// Tool 1: descriptive statistics over a list of numbers
// ---------------------------------------------------------------------------
server.tool(
  "summary_statistics",
  "Compute descriptive statistics (mean, standard deviation, min, max, " +
    "median) for a list of numbers.",
  { values: z.array(z.number()).min(1) },
  async ({ values }) => {
    const n = values.length;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const sorted = [...values].sort((a, b) => a - b);
    const median =
      n % 2 === 0
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        : sorted[(n - 1) / 2];

    const stats = {
      count: n,
      mean,
      std_dev: Math.sqrt(variance),
      min: sorted[0],
      max: sorted[n - 1],
      median,
    };

    return { content: [{ type: "text", text: JSON.stringify(stats, null, 2) }] };
  }
);

// ---------------------------------------------------------------------------
// Tool 2: simple A/B test significance check (two-proportion z-test)
// ---------------------------------------------------------------------------
server.tool(
  "ab_test_significance",
  "Run a two-proportion z-test to check if the difference between two " +
    "conversion rates (A and B) is statistically significant.",
  {
    conversions_a: z.number().int().nonnegative(),
    visitors_a: z.number().int().positive(),
    conversions_b: z.number().int().nonnegative(),
    visitors_b: z.number().int().positive(),
  },
  async ({ conversions_a, visitors_a, conversions_b, visitors_b }) => {
    const rateA = conversions_a / visitors_a;
    const rateB = conversions_b / visitors_b;
    const pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b);
    const se = Math.sqrt(pooled * (1 - pooled) * (1 / visitors_a + 1 / visitors_b));
    const z = se === 0 ? 0 : (rateB - rateA) / se;
    // Two-tailed p-value from the standard normal CDF (Abramowitz & Stegun approximation)
    const pValue = 2 * (1 - normalCdf(Math.abs(z)));

    const result = {
      rate_a: rateA,
      rate_b: rateB,
      z_score: z,
      p_value: pValue,
      significant_at_0_05: pValue < 0.05,
    };

    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  }
);

function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function erf(x: number): number {
  // Abramowitz & Stegun approximation (good to ~1e-7)
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);
  return sign * y;
}

// ---------------------------------------------------------------------------
// Tool 3: lookup from a small research glossary (resource-style tool)
// ---------------------------------------------------------------------------
const GLOSSARY: Record<string, string> = {
  lora: "Low-Rank Adaptation: parameter-efficient fine-tuning via small trainable low-rank matrices.",
  rag: "Retrieval-Augmented Generation: grounding LLM answers in retrieved external documents.",
  mcp: "Model Context Protocol: open standard for connecting AI apps to external tools and data.",
  "bayesian inference": "Updating the probability of a hypothesis as more evidence becomes available.",
};

server.tool(
  "define_term",
  "Look up a short definition of an AI/ML/statistics term from a small glossary.",
  { term: z.string() },
  async ({ term }) => {
    const definition = GLOSSARY[term.trim().toLowerCase()];
    const text = definition ?? `No definition found for "${term}".`;
    return { content: [{ type: "text", text }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("mcp-server-toolkit running on stdio");
}

main().catch((err) => {
  console.error("Fatal error running mcp-server-toolkit:", err);
  process.exit(1);
});
