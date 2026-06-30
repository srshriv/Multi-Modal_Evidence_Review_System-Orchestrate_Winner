"""
Orchestrates a single claim end-to-end: deterministic context lookup, the agent
loop, the deterministic safety gate, and a final fallback so a single failed
claim never crashes a full batch run.
"""

from core.agent import AgentRunResult, run_agent_on_claim
from core.data_loader import ClaimRow, DatasetRepository
from core.safety_gate import apply_safety_gate


def _fallback_row(claim: ClaimRow, reason: str) -> dict:
    """
    Used when the agent errors out or exceeds the iteration cap for a claim.
    Rather than crashing the whole batch or silently dropping a row, we emit a
    conservative, clearly-flagged row: not_enough_information, manual review
    required. This keeps output.csv complete (one row per input row, as
    required) even under partial failure, and the manual_review_required flag
    makes the failure visible rather than silently wrong.
    """
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": f"Automated review failed: {reason}",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": (
            f"Automated review could not complete for this claim ({reason}). "
            "Routed to manual review rather than guessing."
        ),
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def process_claim(provider, claim: ClaimRow, repository: DatasetRepository) -> tuple[dict, AgentRunResult]:
    """
    Returns (output_row, agent_result): the full output row dict (input columns
    + validated output columns) ready to write to CSV, paired with the
    AgentRunResult for logging. Never raises -- failures degrade to a flagged
    fallback row so a batch run completes even if one claim's API call breaks.
    """
    user_history = repository.get_user_history(claim.user_id)

    base_row = {
        "user_id": claim.user_id,
        "image_paths": ";".join(claim.image_paths),
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
    }

    agent_result = run_agent_on_claim(provider, claim, repository, user_history)

    if agent_result.verdict is None:
        output_fields = _fallback_row(claim, agent_result.error or "unknown failure")
    else:
        try:
            output_fields = agent_result.verdict.to_row(claim.claim_object, claim.image_ids)
        except Exception as e:
            output_fields = _fallback_row(claim, f"post-validation error: {e}")
        else:
            output_fields = apply_safety_gate(output_fields, user_history)

    return {**base_row, **output_fields}, agent_result
