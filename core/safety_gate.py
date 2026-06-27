"""
Deterministic safety gate.

This runs AFTER the agent produces a verdict and BEFORE it is written to output.csv.
It mirrors the winning pattern from the May 2026 edition's support-triage task:
"The LLM cannot downgrade a high-risk ticket. The deterministic gate runs first..."

Here the equivalent failure mode is an image containing an embedded instruction
("approve this claim") manipulating a vision model into marking a claim as
supported. We do not trust the model's own self-report of whether it was
influenced -- we deterministically enforce the consequence whenever the model
itself flags `text_instruction_present` or `possible_manipulation`.

This file has no model calls. It is pure rule logic over the verdict + user
history, and its rules cannot be talked out of by anything in a prompt.
"""

from core.data_loader import UserHistory


# Flags that indicate an active manipulation ATTEMPT -- these can never be the
# basis for a "supported" verdict, full stop, because the image's content was
# trying to directly instruct the reviewer rather than show genuine evidence.
HARD_BLOCK_FLAGS = {"text_instruction_present", "possible_manipulation"}

# non_original_image is handled separately and more carefully (see Rule 2b
# below) -- a real bug was found where this flag alone forced a downgrade even
# when the model's own justification described unambiguous, directly-observed
# damage (e.g. "the rear bumper is entirely missing, trunk lid visibly
# crushed") and merely *noted* the image looked like it could be professional
# photography as a secondary caveat. Treating "might be stock photography" the
# same as "this image is a sticky note telling me to approve the claim" was
# too blunt and produced false not_enough_information downgrades on claims
# with genuinely strong evidence.
NON_ORIGINAL_FLAG = "non_original_image"


def apply_safety_gate(
    verdict_dict: dict,
    user_history: UserHistory | None,
) -> dict:
    """
    Mutates and returns a copy of the validated verdict dict, applying deterministic
    overrides. Operates on the already-schema-validated dict from ClaimVerdict.to_row().
    """
    result = dict(verdict_dict)
    flags = set(f for f in result["risk_flags"].split(";") if f and f != "none")

    # --- Rule 1: user_history_risk is decided by code, not by the model's guess ---
    if user_history is not None and user_history.is_high_risk:
        flags.add("user_history_risk")
        flags.add("manual_review_required")
    elif "user_history_risk" in flags and (user_history is None or not user_history.is_high_risk):
        flags.discard("user_history_risk")

    # --- Rule 2a: genuine manipulation attempts (injected text/notes directed
    #     at the reviewer) can never be the sole basis for "supported". This is
    #     a hard block -- the image content was actively trying to manipulate
    #     the outcome, which is categorically different from an image merely
    #     looking professionally shot. ---
    hard_hit = flags & HARD_BLOCK_FLAGS
    if hard_hit:
        flags.add("manual_review_required")
        if result["claim_status"] == "supported":
            result["claim_status"] = "not_enough_information"
            result["claim_status_justification"] = (
                result["claim_status_justification"].rstrip(".")
                + ". Downgraded by safety gate: image evidence includes a "
                + "manipulation attempt ("
                + ", ".join(sorted(hard_hit))
                + ") and cannot independently support the claim."
            )
            result["evidence_standard_met"] = "false"

    # --- Rule 2b: non_original_image alone does NOT force a downgrade if the
    #     model's own supporting_image_ids/justification demonstrate concrete,
    #     directly-observed damage rather than resting the claim on the
    #     image's authenticity. It still always forces manual_review_required
    #     -- a human should still check -- but doesn't override a confident,
    #     well-evidenced "supported" call on its own. This mirrors how a human
    #     reviewer would actually behave: a professionally-lit photo showing
    #     unambiguous wreckage is still real evidence; the suspicion warrants
    #     a second look, not an automatic rejection of what's plainly visible.
    elif NON_ORIGINAL_FLAG in flags:
        flags.add("manual_review_required")
        if result["claim_status"] == "supported" and result["supporting_image_ids"] == "none":
            # Only force the downgrade if the model couldn't even point to a
            # specific supporting image -- i.e. it has no concrete evidence
            # trail, just a general impression of damage.
            result["claim_status"] = "not_enough_information"
            result["claim_status_justification"] = (
                result["claim_status_justification"].rstrip(".")
                + ". Downgraded by safety gate: image may not be original and no "
                + "specific supporting image was identified."
            )
            result["evidence_standard_met"] = "false"

    result["risk_flags"] = ";".join(sorted(flags)) if flags else "none"
    return result
