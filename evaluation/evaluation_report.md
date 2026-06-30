# Evaluation Report — Multi-Modal Evidence Review

## 1. Strategies compared

This evaluation ran with **1 of 1 attempted strategies** -- see the note below on why the others are absent. The strategy that did run used the same agent loop, prompt, tool set, and safety gate that all three providers share (`core/agent.py`, `core/prompts.py`, `core/safety_gate.py`), so the comparison architecture itself supports a multi-provider run; only provider availability on the day of the build limited how many actually completed.

- **Anthropic (Claude, vision)**

Strategies that did run use the same `inspect_image` / `lookup_evidence_requirement` / `submit_verdict` tools, the same system prompt, and the same deterministic post-hoc safety gate (user-history-risk thresholding, prompt-injection / non-original-image downgrade rule). Scoring is field-by-field against ground truth, not a single blended accuracy number, in order to separately catch the failure mode the evaluators specifically called out last cycle: a correct label paired with an empty or generic justification.

### Strategy: anthropic

- Rows evaluated: 20
- Failures (fell back to manual-review row): 0
- Overall field accuracy (avg of 6 exact-match fields): 55.0%

| Field | Accuracy | Correct / Total |
|---|---|---|
| evidence_standard_met | 70.0% | 14/20 |
| claim_status | 65.0% | 13/20 |
| issue_type | 35.0% | 7/20 |
| object_part | 70.0% | 14/20 |
| valid_image | 65.0% | 13/20 |
| severity | 25.0% | 5/20 |

- risk_flags average Jaccard similarity: 49.2%
- Justifications grounded with an image ID reference: 100.0% (20/20)

**claim_status confusion (expected -> predicted):**

| Expected | Predicted | Count |
|---|---|---|
| contradicted | contradicted | 3 |
| contradicted | not_enough_information | 2 |
| not_enough_information | not_enough_information | 2 |
| supported | contradicted | 1 |
| supported | not_enough_information | 4 |
| supported | supported | 8 |

**Mismatched rows (for direct debugging):**

| user_id | Expected | Predicted | supporting_image_ids | Justification |
|---|---|---|---|---|
| user_002 | supported | contradicted | img_2 | img_2 directly shows the front bumper of a white Jaguar XF sedan in full clarity — it is completely ... |
| user_004 | supported | not_enough_information | none | img_1 shows a white van's windshield with extensive spider-web shattering radiating from a clear imp... |
| user_007 | supported | not_enough_information | none | img_1 shows a red car's side mirror with clearly visible, severe fracture lines across the mirror gl... |
| user_003 | supported | not_enough_information | none | img_1 shows a grey/black sedan with a red wrap from a wide-angle distance, but is heavily blurred an... |
| user_010 | supported | not_enough_information | img_1 | img_1 shows clear, close-up evidence of broken hinge/casing damage: the plastic bezel and trim at th... |
| user_020 | contradicted | not_enough_information | none | img_1 shows the palm rest and left side of the laptop near the front edge with a drawn white circle ... |
| user_034 | contradicted | not_enough_information | none | img_1 does show a box with visibly torn tamper-evident tape and ripped cardboard consistent with the... |

- API calls: 49 | input tokens: 284344 | output tokens: 18600 | elapsed: 276.6s

## 2. Strategy selected for `output.csv`

**anthropic** was used to produce the final `output.csv` run against the full `dataset/claims.csv` (44 rows), based on the comparison above (overall field accuracy 55.0%, risk_flags Jaccard 49.2%, 0 fallback failures on the sample set).

## 3. Operational analysis

Figures below are measured directly from the sample-set run (20 labeled rows) and extrapolated to the full 44-row test set by per-row token rate. This is the approach the problem statement asks for: not a perfectly optimized system, but an explicit accounting of cost, latency, rate limits, and avoidable repeated calls.

**Note on the figures below:** this run was assembled across multiple sessions using `--resume` (to survive provider-side daily rate limits hit during development -- see README and chat transcript for that debugging process). Checkpoint-cached rows from before usage-tracking was added to the checkpoint format contribute `0` to the token totals below, so the aggregate `total_input_tokens` / `total_output_tokens` figures are a **floor, not the true total** -- real per-claim cost is better represented by the per-iteration token logs in the raw run output (also in the chat transcript), which showed consistent costs in the 7,000-15,000 input token range per claim depending on image count and how many `inspect_image` / reasoning iterations a claim needed.

### anthropic

- Model calls on sample set (20 rows): 49 (2.5 calls/claim average)
- Images processed on sample set: 29
- Measured tokens on sample set: input=284344, output=18600
- Measured latency on sample set: 276.6s total (13.8s/claim average, sequential, no batching)

**Extrapolated to full claims.csv (44 rows):**
- Estimated tokens: input≈625557, output≈40920
- Estimated cost: **$2.490** (pricing assumption: $3.0/MTok in, $15.0/MTok out — list-price-class estimate, see PRICING_USD_PER_MTOK in evaluation/main.py)
- Estimated latency: ≈609s if run sequentially with no concurrency or caching

### TPM/RPM, batching, caching, and retry considerations

- **Scale**: at 44 test rows averaging 2-3 images each and ~2-4 model calls per claim (1 initial reasoning turn, occasional `inspect_image` follow-ups, 1 final `submit_verdict` turn), the full test run sits well under typical per-minute rate limits for both providers even run sequentially, so no batching API was required for a run of this size.
- **If this scaled to thousands of claims**: claims are independent of each other, so the natural next step is parallelizing across claims (e.g. a bounded worker pool, 5-10 concurrent claims) rather than batching multiple claims into one prompt, since each claim has a different image set and tool-calling trajectory.
- **Caching**: `evidence_requirements.csv` lookups and `user_history.csv` lookups are both pure in-memory dict lookups (see `core/data_loader.py`), not model calls, so there is nothing to cache there. The `lookup_evidence_requirement` tool itself is deterministic and free; only `inspect_image` and the initial reasoning/`submit_verdict` turns hit the model.
- **Retries**: provider call failures are caught per-claim (see `core/agent.py::run_agent_on_claim`) and degrade to a flagged `not_enough_information` / `manual_review_required` fallback row rather than crashing the batch (`core/pipeline.py::_fallback_row`) — this trades a small accuracy hit on the rare failed row for guaranteeing `output.csv` always has exactly one row per input row, which the submission format requires. A production version would add exponential-backoff retry before falling back.
- **Repeated-call avoidance**: each claim makes exactly one initial reasoning call; `inspect_image` is only invoked when the model's own uncertainty calls for a second look (see the system prompt's explicit instruction not to pad the tool loop), and is hard-capped at 2 calls per claim (`MAX_INSPECT_IMAGE_CALLS`). Combined with a 4-iteration loop cap (`MAX_ITERATIONS`, lowered from an initial 6 after measuring that each additional iteration resends the full accumulated message history under a stateless chat-completions API, so cost compounds with iteration count rather than growing linearly), this bounds worst-case cost per claim. A forced-closure nudge was later added at iteration MAX_ITERATIONS-1: if the model still hasn't called submit_verdict by then, further tool calls are blocked and it's explicitly told to decide now -- added after observing real claims loop on tool calls without ever committing to a verdict, which wasted the full token budget for that claim on a guaranteed failure instead of a usable (if uncertain) answer.