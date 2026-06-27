"""
Output schema and allowed-value validation for the Multi-Modal Evidence Review system.

This module is the single source of truth for:
  - the exact output column order required by problem_statement.md
  - the closed sets of allowed values per field
  - validation + coercion logic so a malformed model response never reaches output.csv

Keeping this isolated from the agent/provider code means the allowed-value lists
live in exactly one place, and any prompt that needs them can import from here
instead of duplicating strings that could drift out of sync.
"""

from dataclasses import dataclass, field
from typing import Optional


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_OBJECTS = {"car", "laptop", "package"}

CLAIM_STATUS_VALUES = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE_VALUES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

OBJECT_PART_VALUES = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    },
}

RISK_FLAG_VALUES = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY_VALUES = {"none", "low", "medium", "high", "unknown"}


class SchemaValidationError(ValueError):
    """Raised when a field cannot be coerced into an allowed value at all."""


def _closest_or_unknown(value: str, allowed: set, fallback: str = "unknown") -> str:
    """
    Map a (possibly slightly off) model-produced value onto the allowed set.

    Three tiers, in order:
      1. Exact match.
      2. Case/punctuation-normalized exact match (e.g. "Dent" or "front-bumper").
      3. Substring/keyword match -- a stronger model (Claude in particular) tends
         to produce more descriptive multi-word values than the allowed set's
         single canonical terms, e.g. "rear quarter panel" when the allowed
         value is "quarter_panel", "left headlight" when the allowed value is
         "headlight", or "deep dent" when the allowed value is "dent". A naive
         exact-match-only approach silently collapses all of these to
         "unknown", which was found to be a real, significant cause of
         issue_type/object_part/severity score loss on a Claude run -- the
         model's answers were *more* specific than our schema's vocabulary,
         not wrong, and the validator was punishing that specificity. We match
         by checking whether any allowed value's underscore-split tokens are
         all present as whole words in the input (so "rear quarter panel"
         matches "quarter_panel" via its tokens {"quarter","panel"} both
         appearing), preferring the longest/most-specific allowed value if
         multiple match, to avoid e.g. "headlight" trivially matching almost
         anything containing "light".
    If nothing matches even loosely, fall back to the safe default rather than
    letting a bad string leak into output.csv.
    """
    if value in allowed:
        return value

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    for candidate in allowed:
        if candidate.lower() == normalized:
            return candidate

    # Tier 3: token-subset substring match.
    value_words = set(normalized.split("_"))
    best_match = None
    best_match_token_count = 0
    for candidate in allowed:
        if candidate in ("unknown", "none"):
            continue  # never substring-match onto the fallback values themselves
        candidate_tokens = set(candidate.lower().split("_"))
        if candidate_tokens and candidate_tokens.issubset(value_words):
            if len(candidate_tokens) > best_match_token_count:
                best_match = candidate
                best_match_token_count = len(candidate_tokens)
    if best_match:
        return best_match

    return fallback


def validate_object_part(claim_object: str, part: str) -> str:
    allowed = OBJECT_PART_VALUES.get(claim_object, {"unknown"})
    return _closest_or_unknown(part, allowed, fallback="unknown")


def validate_issue_type(value: str) -> str:
    return _closest_or_unknown(value, ISSUE_TYPE_VALUES, fallback="unknown")


def validate_claim_status(value: str) -> str:
    result = _closest_or_unknown(value, CLAIM_STATUS_VALUES, fallback=None)
    if result is None:
        # claim_status has no safe silent default -- an invalid value here means
        # something is wrong enough that it should force human / retry attention
        # rather than be quietly coerced.
        raise SchemaValidationError(f"Cannot map '{value}' to a valid claim_status")
    return result


def validate_severity(value: str) -> str:
    return _closest_or_unknown(value, SEVERITY_VALUES, fallback="unknown")


def validate_risk_flags(values: list[str]) -> str:
    """Returns the semicolon-joined, deduplicated, validated risk_flags string."""
    if not values:
        return "none"
    cleaned = []
    for v in values:
        mapped = _closest_or_unknown(v, RISK_FLAG_VALUES, fallback=None)
        if mapped and mapped != "none" and mapped not in cleaned:
            cleaned.append(mapped)
    return ";".join(cleaned) if cleaned else "none"


def validate_bool_str(value) -> str:
    """Normalize to the literal strings 'true' / 'false' as required by the CSV schema."""
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip().lower()
    if s in ("true", "yes", "1"):
        return "true"
    if s in ("false", "no", "0"):
        return "false"
    raise SchemaValidationError(f"Cannot coerce '{value}' to true/false")


def validate_image_ids(value, available_image_ids: list[str]) -> str:
    """
    supporting_image_ids must be 'none' or a semicolon-separated list of IDs that
    actually exist in this claim's image set. Anything else (hallucinated IDs,
    free text) is filtered out rather than passed through.
    """
    if value is None:
        return "none"
    if isinstance(value, str):
        raw_ids = [v.strip() for v in value.split(";") if v.strip()]
    else:
        raw_ids = [str(v).strip() for v in value]

    valid = [i for i in raw_ids if i in available_image_ids]
    if not valid:
        return "none"
    # de-dupe, preserve order
    seen = []
    for i in valid:
        if i not in seen:
            seen.append(i)
    return ";".join(seen)


@dataclass
class ClaimVerdict:
    """The structured final answer the agent must produce for one claim."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: str

    def to_row(self, claim_object: str, available_image_ids: list[str]) -> dict:
        """Validate every field and produce the final CSV row dict (minus input columns)."""
        return {
            "evidence_standard_met": validate_bool_str(self.evidence_standard_met),
            "evidence_standard_met_reason": str(self.evidence_standard_met_reason).strip(),
            "risk_flags": validate_risk_flags(self.risk_flags),
            "issue_type": validate_issue_type(self.issue_type),
            "object_part": validate_object_part(claim_object, self.object_part),
            "claim_status": validate_claim_status(self.claim_status),
            "claim_status_justification": str(self.claim_status_justification).strip(),
            "supporting_image_ids": validate_image_ids(self.supporting_image_ids, available_image_ids),
            "valid_image": validate_bool_str(self.valid_image),
            "severity": validate_severity(self.severity),
        }


# The JSON schema handed to both providers for the forced final tool call.
# Identical across providers so the comparison in evaluation/ is apples-to-apples.
SUBMIT_VERDICT_TOOL_SCHEMA = {
    "name": "submit_verdict",
    "description": (
        "Submit the final structured verdict for this claim. Call this only once you "
        "have inspected the available images and have enough information to decide, "
        "or once you have concluded the evidence is insufficient. This ends the review."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "evidence_standard_met": {
                "type": "boolean",
                "description": (
                    "True if the image set is sufficient to evaluate the claim, "
                    "false otherwise. Must be a JSON boolean (true or false, no quotes), "
                    "never the string \"true\" or \"false\"."
                ),
            },
            "evidence_standard_met_reason": {
                "type": "string",
                "description": "Short reason for the evidence-sufficiency decision.",
            },
            "risk_flags": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(RISK_FLAG_VALUES)},
                "description": "Any applicable risk flags. Use [] if none apply.",
            },
            "issue_type": {
                "type": "string",
                "enum": sorted(ISSUE_TYPE_VALUES),
            },
            "object_part": {
                "type": "string",
                "description": "Relevant object part. Must match the claim_object's allowed part list.",
            },
            "claim_status": {
                "type": "string",
                "enum": sorted(CLAIM_STATUS_VALUES),
            },
            "claim_status_justification": {
                "type": "string",
                "description": (
                    "Concise, image-grounded explanation. Reference specific image IDs "
                    "(e.g. img_1) where helpful. Must not be generic -- explain what is "
                    "actually visible."
                ),
            },
            "supporting_image_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Image IDs (e.g. 'img_1') that support the decision. Empty array if none.",
            },
            "valid_image": {
                "type": "boolean",
                "description": (
                    "True if the image set is usable for automated review at all, "
                    "false otherwise. Must be a JSON boolean (true or false, no quotes), "
                    "never the string \"true\" or \"false\"."
                ),
            },
            "severity": {
                "type": "string",
                "enum": sorted(SEVERITY_VALUES),
            },
        },
        "required": [
            "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
            "issue_type", "object_part", "claim_status", "claim_status_justification",
            "supporting_image_ids", "valid_image", "severity",
        ],
    },
}
