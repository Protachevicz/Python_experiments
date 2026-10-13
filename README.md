# Python Experiments

A collection of applied AI/ML, GenAI/Agents, and statistics experiments, organized by subject into self-contained folders, each with its own README explaining the project, how it works, and how to run it. Most projects are in Python; one (the MCP server) is in TypeScript, chosen because that's the ecosystem its tooling targets.

## Projects

| Folder | Topic | Projects |
|---|---|---|
| [`ai-agent-tool-calling/`](ai-agent-tool-calling) | AI Agents | ReAct-style agent loop with tool calling (calculator, time, knowledge lookup) |
| [`rag-pipeline-chroma/`](rag-pipeline-chroma) | Retrieval-Augmented Generation | Chunking, local embeddings, ChromaDB vector search, grounded LLM answers |
| [`mcp-server-toolkit/`](mcp-server-toolkit) | Model Context Protocol (TypeScript) | A from-scratch MCP server exposing statistics, A/B testing, and glossary tools |
| [`experiment-design-ab-testing/`](experiment-design-ab-testing) | A/B Testing & Experimentation | Significance over time (video), segmented conditions, 50/50 convergence |
| [`teacher-student-distillation/`](teacher-student-distillation) | Teacher-Student Model Distillation | Professor/student distillation pipeline, BERT-tokenized variant, schema diagram |
| [`deepseek-r1-lora-finetuning/`](deepseek-r1-lora-finetuning) | LLM Fine-Tuning | DeepSeek-R1 LoRA/QLoRA fine-tuning on a medical reasoning dataset |
| [`central-limit-theorem-sampling/`](central-limit-theorem-sampling) | Statistical Inference | Central Limit Theorem demo on real EPA air quality data |
| [`music-taste-recommender/`](music-taste-recommender) | Recommender Systems | Content-based music recommender simulation |
| [`temporal-lobe-epilepsy-intermittency/`](temporal-lobe-epilepsy-intermittency) | Computational Neuroscience (C) | Code for *Epilepsy & Behavior* 2023: IAIN connectivity, intermittent synchronization, optogenetic control |
| [`short-term-plasticity-analytical-solution/`](short-term-plasticity-analytical-solution) | Computational Neuroscience (C) | Code for *Chaos, Solitons & Fractals* 2024: closed-form analytical solution to short-term synaptic plasticity |
| [`seasonal-infectious-disease-chaos/`](seasonal-infectious-disease-chaos) | Epidemiology (C) | Code for *Chaos, Solitons & Fractals* 2023: bistable chaotic/periodic dynamics in a seasonally forced SEIRS model |

Each folder's README explains the purpose, methodology, and requirements for every script inside it.

## Related research-code repositories

A few other published-paper codebases were originally hosted at [henon.if.usp.br/OscilControlData](http://henon.if.usp.br/OscilControlData/) and now live in their own dedicated repositories:

- [`Chimera-and-Plasticity`](https://github.com/Protachevicz/Chimera-and-Plasticity) — *AIP Advances* 2022, chimera states induced by STDP
- [`Synaptic_potentiation_dependence`](https://github.com/Protachevicz/Synaptic_potentiation_dependence) — *EPJ Special Topics*, synaptic potentiation dependence on spike variability
- [`STDP_Delay`](https://github.com/Protachevicz/STDP_Delay) — *Chaos, Solitons & Fractals* 2023, plastic networks with transmission delays
