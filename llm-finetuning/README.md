# LLM Fine-Tuning: DeepSeek-R1 with LoRA/QLoRA

### `finetune_deepseek_r1_lora.py`
Fine-tunes a DeepSeek-R1 distilled model (`DeepSeek-R1-Distill-Qwen-1.5B`, with `DeepSeek-R1-Distill-Llama-8B` as a documented alternative) on the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset, a medical chain-of-thought reasoning dataset.

**Pipeline:**
1. Loads and splits the dataset (90% train / 10% test)
2. Tokenizes the `Question` field with the model's own tokenizer
3. Loads the base model in **4-bit quantization (QLoRA)** via `BitsAndBytesConfig` (nf4, double quantization, float16 compute) to fit large models in limited GPU memory
4. Wraps the model with a **LoRA adapter** (`peft`) so only a small fraction of parameters are trained
5. Fine-tunes with Hugging Face's `Trainer`, evaluating and checkpointing every epoch
6. Saves the fine-tuned adapter and tokenizer to `./fine_tuned_deepseek`

**Purpose:** a reproducible, memory-efficient recipe for adapting a reasoning-focused open LLM to a specialized domain (medical QA) using parameter-efficient fine-tuning, the same family of techniques used in the author's `deepseek-r1-medical-finetune` project (which uses the Unsloth library for faster/lighter training). This script shows the equivalent workflow with the standard Hugging Face `transformers` + `peft` + `bitsandbytes` stack.

**Requirements:** `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`. Requires a CUDA-capable GPU.

**Run:** `python finetune_deepseek_r1_lora.py`
