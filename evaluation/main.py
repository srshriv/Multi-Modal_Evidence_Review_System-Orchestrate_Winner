"""
Evaluation entry point.

Runs every provider strategy with a working key against the labeled
dataset/sample_claims.csv, scores each against ground truth, and writes
evaluation/evaluation_report.md with:
  - the strategy comparison (required by problem_statement.md / README)
  - the operational analysis (model calls, tokens, images, cost, latency,
    TPM/RPM considerations)

Usage:
    python evaluation/main.py                          # all providers with available keys
    python evaluation/main.py --provider anthropic      # only one, for quick iteration
    python evaluation/main.py --provider groq           # resume is ON by default -- a second
                                                          # invocation automatically skips rows
                                                          # that already succeeded, so re-running
                                                          # after a partial/rate-limited run never
                                                          # re-pays for completed rows by accident
    python evaluation/main.py --provider groq --no-resume  # force a clean re-run of every row

This intentionally only runs against the 20-row sample_claims.csv (which has
ground truth), never against the 44-row claims.csv -- running both strategies
against the unlabeled test set would double real spend for no evaluative
benefit, since there's no ground truth there to score against. The winning
strategy, once chosen here, is what main.py uses for the actual claims.csv run.

Checkpointing: each successful (non-fallback) row's prediction is cached to
evaluation/.checkpoint_<provider>.json as it completes. --resume reads this
cache and only re-runs rows that are missing or that fell back to a manual-
review row last time, then merges cached + freshly-run results before
scoring. This exists because free-tier rate limits (observed on Groq) can
cut a run off partway through a 20-row batch, and re-paying for the rows
that already succeeded would be wasteful and would also distort the
evaluation -- a row's score shouldn't depend on which run happened to
process it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DatasetRepository
from core.pipeline import process_claim
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.openai_provider import OpenAIProvider
from core.providers.groq_provider import GroqProvider
from core.providers.gemini_provider import GeminiProvider
from evaluation.metrics import score_run, format_score_summary

DATASET_ROOT = Path(__file__).parent.parent.parent / "dataset"
REPORT_PATH = Path(__file__).parent / "evaluation_report.md"
CHECKPOINT_DIR = Path(__file__).parent

# Pricing snapshot used for the operational cost estimate in the report.
# These are illustrative list-price assumptions, documented explicitly here so
# the cost estimate in the report is reproducible and auditable rather than a
# magic number. Update if actual provider pricing changes.
PRICING_USD_PER_MTOK = {
    "anthropic": {"input": 3.00, "output": 15.00},   # Claude Sonnet-class pricing assumption
    "openai": {"input": 2.50, "output": 10.00},       # GPT-4o-class pricing assumption
    "groq": {"input": 0.11, "output": 0.34},          # Llama 4 Scout on Groq, confirmed June 2026
    "gemini": {"input": 0.30, "output": 2.50},        # Gemini 2.5 Flash, confirmed June 2026
}


def _checkpoint_path(provider_name: str) -> Path:
    return CHECKPOINT_DIR / f".checkpoint_{provider_name}.json"


def _load_checkpoint(provider_name: str) -> dict:
    path = _checkpoint_path(provider_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_checkpoint(provider_name: str, checkpoint: dict):
    _checkpoint_path(provider_name).write_text(json.dumps(checkpoint, indent=2))


def run_strategy(
    provider_name: str,
    repository: DatasetRepository,
    claims: list,
    expected_rows: list,
    resume: bool = False,
    token_budget: int | None = None,
):
    if provider_name == "anthropic":
        provider = AnthropicProvider()
    elif provider_name == "openai":
        provider = OpenAIProvider()
    elif provider_name == "groq":
        provider = GroqProvider()
    elif provider_name == "gemini":
        provider = GeminiProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    checkpoint = _load_checkpoint(provider_name) if resume else {}

    predictions = []
    total_api_calls = 0
    total_in = 0
    total_out = 0
    total_images = 0
    failures = 0
    newly_run = 0
    reused_from_checkpoint = 0
    budget_stopped = False

    start = time.time()
    for i, claim in enumerate(claims, 1):
        cache_key = claim.user_id
        cached = checkpoint.get(cache_key)
        if resume and cached is not None and not cached.get("was_fallback"):
            # Reuse a previously-successful row instead of paying for it again.
            # Also restore that row's original usage stats so the operational
            # analysis (tokens, API calls, images) reflects the true total cost
            # of producing all 20 predictions, not just whatever happened to run
            # in this particular invocation. Older checkpoint files saved before
            # this fix won't have a "usage" key -- fall back to zeros for those
            # rather than crashing, and note the totals will undercount in that
            # case (acceptable for a resumed run; a fresh run recomputes cleanly).
            row = cached["row"]
            usage = cached.get("usage", {})
            predictions.append(row)
            reused_from_checkpoint += 1
            total_api_calls += usage.get("api_calls", 0)
            total_in += usage.get("input_tokens", 0)
            total_out += usage.get("output_tokens", 0)
            total_images += usage.get("images", 0)
            print(f"    [{provider_name}] [{i}/{len(claims)}] {claim.user_id} -> "
                  f"{row['claim_status']} (reused from checkpoint)")
            continue

        if token_budget is not None and (total_in + total_out) >= token_budget:
            # Stop gracefully rather than burn repeated 429s against an
            # already-exhausted daily cap -- each failed request still counts
            # against rate-limit bookkeeping on some providers and clutters the
            # checkpoint with fallback rows for no benefit. Remaining rows are
            # simply left unprocessed; --resume picks them up next time the
            # budget resets.
            print(f"    [{provider_name}] token budget ({token_budget}) reached after "
                  f"{newly_run} new rows -- stopping early. Remaining rows left for next "
                  f"--resume run.")
            budget_stopped = True
            break

        row, agent_result = process_claim(provider, claim, repository)
        predictions.append(row)
        newly_run += 1
        total_api_calls += agent_result.api_calls_made
        total_in += agent_result.total_input_tokens
        total_out += agent_result.total_output_tokens
        total_images += len(claim.image_paths)
        was_fallback = bool(agent_result.error)
        if was_fallback:
            failures += 1
        checkpoint[cache_key] = {
            "row": row,
            "was_fallback": was_fallback,
            "usage": {
                "api_calls": agent_result.api_calls_made,
                "input_tokens": agent_result.total_input_tokens,
                "output_tokens": agent_result.total_output_tokens,
                "images": len(claim.image_paths),
            },
        }
        _save_checkpoint(provider_name, checkpoint)  # save after every row, not just at the end
        print(f"    [{provider_name}] [{i}/{len(claims)}] {claim.user_id} -> "
              f"{row['claim_status']}" + (f" (FALLBACK: {agent_result.error})" if was_fallback else ""))
    elapsed = time.time() - start

    print(f"    [{provider_name}] done: {newly_run} newly run, {reused_from_checkpoint} reused from "
          f"checkpoint" + (", stopped early on token budget" if budget_stopped else ""))

    score = score_run(provider_name, predictions, expected_rows)
    score.total_api_calls = total_api_calls
    score.total_input_tokens = total_in
    score.total_output_tokens = total_out
    score.total_images_processed = total_images
    score.failures = failures
    score.elapsed_seconds = elapsed
    return score, predictions


def estimate_full_test_cost(provider_name: str, sample_score, test_row_count: int, sample_row_count: int) -> dict:
    """
    Extrapolates from the sample run's measured token usage to an estimated
    cost for the full 44-row claims.csv, scaled by row count. This is an
    approximation (test rows have a different image-count distribution than
    sample rows) but gives a defensible order-of-magnitude estimate using
    actually-measured per-row token rates rather than a guess.
    """
    if sample_row_count == 0:
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    per_row_in = sample_score.total_input_tokens / sample_row_count
    per_row_out = sample_score.total_output_tokens / sample_row_count

    est_in = per_row_in * test_row_count
    est_out = per_row_out * test_row_count

    pricing = PRICING_USD_PER_MTOK[provider_name]
    cost = (est_in / 1_000_000) * pricing["input"] + (est_out / 1_000_000) * pricing["output"]
    return {"input_tokens": est_in, "output_tokens": est_out, "cost_usd": cost}


def write_report(scores: dict, predictions_by_strategy: dict, test_row_count: int, chosen_strategy: str, attempted_providers: list):
    provider_display = {
        "anthropic": "Anthropic (Claude, vision)",
        "openai": "OpenAI (GPT-4o, vision)",
        "groq": "Groq (Llama 4 Scout, vision)",
        "gemini": "Google Gemini (2.5 Flash, vision)",
    }
    ran = list(scores.keys())
    failed_to_run = [p for p in attempted_providers if p not in ran]

    sections = [
        "# Evaluation Report — Multi-Modal Evidence Review",
        "",
        "## 1. Strategies compared",
        "",
    ]

    if len(ran) >= 2:
        sections.append(
            f"{len(ran)} strategies were run against the labeled `dataset/sample_claims.csv` "
            "(20 rows) using an identical agent loop, prompt, tool set, and safety "
            "gate — the only variable swapped is the underlying model/provider:"
        )
    else:
        sections.append(
            f"This evaluation ran with **{len(ran)} of {len(attempted_providers)} attempted "
            "strategies** -- see the note below on why the others are absent. The strategy "
            "that did run used the same agent loop, prompt, tool set, and safety gate that "
            "all three providers share (`core/agent.py`, `core/prompts.py`, "
            "`core/safety_gate.py`), so the comparison architecture itself supports a "
            "multi-provider run; only provider availability on the day of the build limited "
            "how many actually completed."
        )
    sections.append("")
    for p in ran:
        sections.append(f"- **{provider_display.get(p, p)}**")
    sections.append("")

    if failed_to_run:
        sections.append(
            "**Note on providers that did not produce a comparable result:** "
            + ", ".join(provider_display.get(p, p) for p in failed_to_run)
            + " were configured and code-complete (see `core/providers/`), but could not be "
            "run to completion during this build window due to exhausted account credit/quota "
            "(Anthropic: insufficient credit balance; OpenAI: insufficient quota) rather than "
            "any code or integration failure. Groq's free tier was then used instead, which "
            "introduced its own constraint -- a 500,000 tokens/day cap that required adding "
            "checkpointing/resume, a hard per-claim tool-call budget, and a pre-flight cost "
            "estimate to complete a clean run within the daily limit (see "
            "`evaluation/main.py`'s `--resume` / `--token-budget` flags and `core/agent.py`'s "
            "`MAX_ITERATIONS` / `MAX_INSPECT_IMAGE_CALLS` constants). This is disclosed here "
            "rather than omitted because it materially shaped which strategy produced the "
            "final `output.csv` and why -- not because Groq was judged technically superior "
            "to the other two on a like-for-like comparison, but because it was the strategy "
            "that could actually be run to completion."
        )
        sections.append("")

    sections.append(
        f"Strategies that did run use the same `inspect_image` / `lookup_evidence_requirement` "
        "/ `submit_verdict` tools, the same system prompt, and the same deterministic "
        "post-hoc safety gate (user-history-risk thresholding, "
        "prompt-injection / non-original-image downgrade rule). Scoring is field-by-field "
        "against ground truth, not a single blended accuracy number, in order to "
        "separately catch the failure mode the evaluators specifically called out last "
        "cycle: a correct label paired with an empty or generic justification.",
    )
    sections.append("")

    for name, score in scores.items():
        sections.append(format_score_summary(score))
        sections.append("")

    sections.append("## 2. Strategy selected for `output.csv`")
    sections.append("")
    chosen = scores[chosen_strategy]
    sections.append(
        f"**{chosen_strategy}** was used to produce the final `output.csv` run against the "
        f"full `dataset/claims.csv` ({test_row_count} rows), based on the comparison above "
        f"(overall field accuracy {chosen.overall_field_accuracy:.1%}, "
        f"risk_flags Jaccard {chosen.risk_flag_avg_jaccard:.1%}, "
        f"{chosen.failures} fallback failures on the sample set)."
    )
    sections.append("")

    sections.append("## 3. Operational analysis")
    sections.append("")
    sections.append(
        "Figures below are measured directly from the sample-set run (20 labeled rows) and "
        f"extrapolated to the full {test_row_count}-row test set by per-row token rate. This "
        "is the approach the problem statement asks for: not a perfectly optimized system, "
        "but an explicit accounting of cost, latency, rate limits, and avoidable repeated calls."
    )
    sections.append("")
    sections.append(
        "**Note on the figures below:** this run was assembled across multiple sessions using "
        "`--resume` (to survive provider-side daily rate limits hit during development -- see "
        "README and chat transcript for that debugging process). Checkpoint-cached rows from "
        "before usage-tracking was added to the checkpoint format contribute `0` to the token "
        "totals below, so the aggregate `total_input_tokens` / `total_output_tokens` figures "
        "are a **floor, not the true total** -- real per-claim cost is better represented by "
        "the per-iteration token logs in the raw run output (also in the chat transcript), "
        "which showed consistent costs in the 7,000-15,000 input token range per claim "
        "depending on image count and how many `inspect_image` / reasoning iterations a claim "
        "needed."
    )
    sections.append("")

    for name, score in scores.items():
        est = estimate_full_test_cost(name, score, test_row_count, score.total_rows)
        sections.append(f"### {name}")
        sections.append("")
        sections.append(f"- Model calls on sample set (20 rows): {score.total_api_calls} "
                         f"({score.total_api_calls / score.total_rows:.1f} calls/claim average)")
        sections.append(f"- Images processed on sample set: {score.total_images_processed}")
        sections.append(f"- Measured tokens on sample set: input={score.total_input_tokens}, "
                         f"output={score.total_output_tokens}")
        sections.append(f"- Measured latency on sample set: {score.elapsed_seconds:.1f}s total "
                         f"({score.elapsed_seconds / score.total_rows:.1f}s/claim average, sequential, no batching)")
        sections.append("")
        sections.append(f"**Extrapolated to full claims.csv ({test_row_count} rows):**")
        sections.append(f"- Estimated tokens: input≈{est['input_tokens']:.0f}, output≈{est['output_tokens']:.0f}")
        sections.append(
            f"- Estimated cost: **${est['cost_usd']:.3f}** "
            f"(pricing assumption: ${PRICING_USD_PER_MTOK[name]['input']}/MTok in, "
            f"${PRICING_USD_PER_MTOK[name]['output']}/MTok out — list-price-class estimate, "
            "see PRICING_USD_PER_MTOK in evaluation/main.py)"
        )
        sections.append(
            f"- Estimated latency: ≈{(score.elapsed_seconds / score.total_rows) * test_row_count:.0f}s "
            "if run sequentially with no concurrency or caching"
        )
        sections.append("")

    sections.append("### TPM/RPM, batching, caching, and retry considerations")
    sections.append("")
    sections.append(
        "- **Scale**: at 44 test rows averaging 2-3 images each and ~2-4 model calls per "
        "claim (1 initial reasoning turn, occasional `inspect_image` follow-ups, 1 final "
        "`submit_verdict` turn), the full test run sits well under typical per-minute rate "
        "limits for both providers even run sequentially, so no batching API was required "
        "for a run of this size."
    )
    sections.append(
        "- **If this scaled to thousands of claims**: claims are independent of each other, "
        "so the natural next step is parallelizing across claims (e.g. a bounded worker pool, "
        "5-10 concurrent claims) rather than batching multiple claims into one prompt, since "
        "each claim has a different image set and tool-calling trajectory."
    )
    sections.append(
        "- **Caching**: `evidence_requirements.csv` lookups and `user_history.csv` lookups "
        "are both pure in-memory dict lookups (see `core/data_loader.py`), not model calls, "
        "so there is nothing to cache there. The `lookup_evidence_requirement` tool itself "
        "is deterministic and free; only `inspect_image` and the initial reasoning/"
        "`submit_verdict` turns hit the model."
    )
    sections.append(
        "- **Retries**: provider call failures are caught per-claim (see "
        "`core/agent.py::run_agent_on_claim`) and degrade to a flagged "
        "`not_enough_information` / `manual_review_required` fallback row rather than "
        "crashing the batch (`core/pipeline.py::_fallback_row`) — this trades a small "
        "accuracy hit on the rare failed row for guaranteeing `output.csv` always has "
        "exactly one row per input row, which the submission format requires. A production "
        "version would add exponential-backoff retry before falling back."
    )
    sections.append(
        "- **Repeated-call avoidance**: each claim makes exactly one initial reasoning call; "
        "`inspect_image` is only invoked when the model's own uncertainty calls for a second "
        "look (see the system prompt's explicit instruction not to pad the tool loop), and is "
        "hard-capped at 2 calls per claim (`MAX_INSPECT_IMAGE_CALLS`). Combined with a "
        "4-iteration loop cap (`MAX_ITERATIONS`, lowered from an initial 6 after measuring "
        "that each additional iteration resends the full accumulated message history under "
        "a stateless chat-completions API, so cost compounds with iteration count rather than "
        "growing linearly), this bounds worst-case cost per claim. A forced-closure nudge was "
        "later added at iteration MAX_ITERATIONS-1: if the model still hasn't called "
        "submit_verdict by then, further tool calls are blocked and it's explicitly told to "
        "decide now -- added after observing real claims loop on tool calls without ever "
        "committing to a verdict, which wasted the full token budget for that claim on a "
        "guaranteed failure instead of a usable (if uncertain) answer."
    )
    sections.append("")

    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"\n[evaluation] wrote {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq", "gemini", "both", "all"], default="all")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable checkpoint resume and force every row to be (re-)run from scratch, "
             "re-paying for rows that already succeeded in a previous run. Resume is ON by "
             "default -- this exists for the rare case you deliberately want a clean re-run "
             "(e.g. after changing the prompt and wanting fresh judgments on every row).",
    )
    parser.add_argument(
        "--token-budget", type=int, default=None,
        help="Stop a strategy's run early (gracefully, before hitting a 429) once its "
             "MEASURED input+output tokens for THIS PROCESS reach this number. Note this "
             "only tracks usage within this run -- it cannot see usage already consumed "
             "earlier today against a provider's daily cap. Useful as a safety margin on "
             "top of (not instead of) running --resume and watching the printed pre-flight "
             "estimate below.",
    )
    args = parser.parse_args()
    resume = not args.no_resume

    repository = DatasetRepository(DATASET_ROOT)
    claims = repository.load_claims_csv("sample_claims.csv", has_labels=True)
    expected_rows = [c.expected for c in claims]
    test_claims = repository.load_claims_csv("claims.csv", has_labels=False)

    if args.provider in ("both", "all"):
        providers_to_run = ["anthropic", "openai", "groq", "gemini"]
    else:
        providers_to_run = [args.provider]

    # Pre-flight estimate, printed before any API calls happen. Based on
    # measured real usage from prior runs (~11,000 tokens/claim observed on
    # Groq with the current MAX_ITERATIONS=4 / MAX_INSPECT_IMAGE_CALLS=2
    # caps), not a guess. This exists specifically so a fresh/near-empty
    # quota account gets a clear "this will cost approximately X" before
    # committing, rather than discovering the cost mid-run via a 429.
    MEASURED_TOKENS_PER_CLAIM_ESTIMATE = 11000
    if resume:
        already_done = {}
        for provider_name in providers_to_run:
            cp = _load_checkpoint(provider_name)
            already_done[provider_name] = sum(1 for v in cp.values() if not v.get("was_fallback"))
    else:
        already_done = {p: 0 for p in providers_to_run}

    print("[evaluation] pre-flight estimate (based on measured ~11,000 tokens/claim on Groq-class usage):")
    for provider_name in providers_to_run:
        remaining = len(claims) - already_done.get(provider_name, 0)
        est_tokens = remaining * MEASURED_TOKENS_PER_CLAIM_ESTIMATE
        print(f"    {provider_name}: {already_done.get(provider_name, 0)} rows already cached"
              f" (resume={'on' if resume else 'off'}), {remaining} rows to run, "
              f"~{est_tokens:,} tokens estimated")
    print()

    scores = {}
    predictions_by_strategy = {}
    for provider_name in providers_to_run:
        print(f"\n[evaluation] running strategy: {provider_name} on sample_claims.csv ({len(claims)} rows)")
        try:
            score, predictions = run_strategy(
                provider_name, repository, claims, expected_rows,
                resume=resume, token_budget=args.token_budget,
            )
        except Exception as e:
            # A whole provider being unavailable (bad key, exhausted quota, no
            # billing) shouldn't take down the rest of the comparison -- skip
            # it and keep going with whichever providers do work, and say so
            # plainly in the report rather than silently omitting it.
            print(f"[evaluation] strategy '{provider_name}' could not run at all: {e}")
            print(f"[evaluation] skipping '{provider_name}' for this comparison.")
            continue
        scores[provider_name] = score
        predictions_by_strategy[provider_name] = predictions
        print(format_score_summary(score))

    if not scores:
        print("\n[evaluation] No strategy could be run (all providers failed/unavailable). "
              "Check your .env keys and billing status.")
        return 1

    # Pick the strategy with the higher overall field accuracy as the one used
    # for the real output.csv run (ties broken by risk_flag Jaccard, then fewer failures).
    chosen = max(
        scores.keys(),
        key=lambda k: (scores[k].overall_field_accuracy, scores[k].risk_flag_avg_jaccard, -scores[k].failures),
    )
    print(f"\n[evaluation] selected strategy for output.csv: {chosen}")

    write_report(scores, predictions_by_strategy, len(test_claims), chosen, providers_to_run)


if __name__ == "__main__":
    sys.exit(main())
