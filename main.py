"""
Main entry point for the Multi-Modal Evidence Review system.

Usage:
    python main.py                              # uses default provider (anthropic), full claims.csv
    python main.py --provider openai
    python main.py --provider anthropic --limit 5   # smoke-test on first 5 rows
    python main.py --input ../dataset/sample_claims.csv --output sample_output.csv

Reads ANTHROPIC_API_KEY / OPENAI_API_KEY from the environment (via .env, loaded
through python-dotenv). Never hardcode keys.
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from core.data_loader import DatasetRepository, write_output_csv
from core.pipeline import process_claim
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.openai_provider import OpenAIProvider
from core.providers.groq_provider import GroqProvider
from core.providers.gemini_provider import GeminiProvider
from core.schema import OUTPUT_COLUMNS

DEFAULT_DATASET_ROOT = Path(__file__).parent.parent / "dataset"


def get_provider(name: str):
    if name == "anthropic":
        return AnthropicProvider()
    elif name == "openai":
        return OpenAIProvider()
    elif name == "groq":
        return GroqProvider()
    elif name == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unknown provider: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run the claim review agent over a claims CSV.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq", "gemini"], default="anthropic")
    parser.add_argument(
        "--dataset-root", default=str(DEFAULT_DATASET_ROOT),
        help="Path to the dataset/ directory (contains claims.csv, images/, etc.)",
    )
    parser.add_argument("--input", default="claims.csv", help="Filename within dataset-root to read.")
    parser.add_argument("--output", default="output.csv", help="Where to write predictions.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (debug).")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    repository = DatasetRepository(dataset_root)
    claims = repository.load_claims_csv(args.input, has_labels=False)

    if args.limit:
        claims = claims[: args.limit]

    print(f"[main] provider={args.provider} input={args.input} rows={len(claims)}")

    provider = get_provider(args.provider)

    output_rows = []
    total_api_calls = 0
    total_in_tokens = 0
    total_out_tokens = 0
    failures = 0
    start = time.time()

    for i, claim in enumerate(claims, 1):
        row, agent_result = process_claim(provider, claim, repository)
        output_rows.append(row)
        total_api_calls += agent_result.api_calls_made
        total_in_tokens += agent_result.total_input_tokens
        total_out_tokens += agent_result.total_output_tokens
        if agent_result.error:
            failures += 1
            print(f"  [{i}/{len(claims)}] {claim.user_id} -> FALLBACK ({agent_result.error})")
        else:
            print(
                f"  [{i}/{len(claims)}] {claim.user_id} -> "
                f"{row['claim_status']} (iters={agent_result.iterations_used}, "
                f"tools={agent_result.tool_calls_made})"
            )

    elapsed = time.time() - start
    output_path = Path(args.output)  # written relative to cwd, matching the README's expectation
    write_output_csv(output_rows, output_path, OUTPUT_COLUMNS)

    print()
    print(f"[main] done in {elapsed:.1f}s | rows={len(claims)} | api_calls={total_api_calls} "
          f"| failures={failures}")
    print(f"[main] tokens: in={total_in_tokens} out={total_out_tokens}")
    print(f"[main] wrote {output_path}")


if __name__ == "__main__":
    sys.exit(main())
