# Multi-Modal Evidence Review

> 🏆 **#1 globally — HackerRank Orchestrate June 2026** · 15,295 registered · 2,039 shipped

A single-agent system that verifies insurance damage claims using submitted images, a claim conversation, user history, and minimum evidence requirements. For each claim, the agent outputs a structured verdict: `supported`, `contradicted`, or `not_enough_information`.

---

## Architecture

Single agent with tools — one model drives a bounded tool-calling loop per claim. Three tools: `inspect_image` (focused re-look at a specific image), `lookup_evidence_requirement` (deterministic reference lookup), and `submit_verdict` (forced structured output). All images are attached upfront; the model decides when it has enough to commit.

```
claims.csv + images + user_history.csv + evidence_reqs.csv
        │
        ▼
Deterministic loading (no model calls)
        │
        ▼
Agentic tool loop — core/agent.py
  · inspect_image       (model-driven, capped at 2/claim)
  · lookup_evidence_req (deterministic)
  · submit_verdict      (model decides when done; hard cap: 4 iterations)
        │
        ▼
Schema validation — token-subset matching for rich model output
        │
        ▼
Deterministic safety gate — core/safety_gate.py
  · injection attempts  → hard block regardless of model verdict
  · non-original images → flag + manual review, not automatic downgrade
  · user history risk   → computed in code, not by model
        │
        ▼
output.csv
```

---

## Key Decisions

**Safety gate in code, not prompt.** Prompt instructions can be overridden by adversarial image content. The gate enforces post-model rules that cannot be argued away — injection attempts hard-block a `supported` verdict; history risk flags are computed deterministically.

**Images upfront, `inspect_image` for uncertainty.** A model can't decide which image to fetch first without seeing any of them. All images attach in turn one; `inspect_image` is a genuine model-driven decision, not a scripted sequence.

**Token-subset schema matching.** Stronger models produce richer field values (`"rear quarter panel"` vs `"quarter_panel"`). Naive exact matching silently collapses these to `"unknown"`. A three-tier matcher (exact → normalized → token-subset) handles this correctly.

**Provider-agnostic interface.** All four providers share the same `batch_tool_results` / `build_image_blocks` / `run_turn` interface. Anthropic requires all tool results batched into one message per turn; OpenAI/Groq accept separate messages. This difference is handled at the provider level, invisibly to the agent loop.

---

## Adversarial Cases Handled

| Pattern | Flag | Action |
|---|---|---|
| In-image instruction ("approve this claim") | `text_instruction_present` | Hard block |
| Stock photo watermark | `non_original_image` | Manual review required |
| Wrong object (toy car, food can) | `wrong_object` | `contradicted` |
| Two different vehicles in one claim | `claim_mismatch` | Cross-image reconciliation before verdict |
| Prompt injection in claim text | noted in justification | No influence on verdict |

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add whichever keys you have
```

## Running

```bash
# Evaluate on labeled sample set first
python evaluation/main.py --provider anthropic

# Run on full test set
python main.py --provider anthropic --output ../output.csv

# Debug mode
AGENT_VERBOSE=1 python evaluation/main.py --provider anthropic
```

---

## Project Structure

```
code/
├── main.py
├── core/
│   ├── agent.py          # agentic loop
│   ├── schema.py         # validation + token-subset matching
│   ├── safety_gate.py    # deterministic post-model rules
│   ├── pipeline.py       # end-to-end orchestration + fallback
│   ├── prompts.py        # system prompt
│   ├── tools.py          # tool schemas + deterministic execution
│   ├── data_loader.py    # CSV + image loading
│   └── providers/        # anthropic · openai · groq · gemini
└── evaluation/
    ├── main.py           # multi-provider comparison + scoring
    ├── metrics.py        # field-by-field metrics + mismatch table
    └── evaluation_report.md
```

---

## Providers

| Provider | Model |
|---|---|
| Anthropic | `claude-sonnet-4-6` |
| OpenAI | `gpt-4o` |
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Gemini | `gemini-2.5-flash` (uses `google-genai`, not deprecated `google-generativeai`) |

All four implement the same interface. Evaluation skips any provider with a missing or exhausted key rather than crashing.