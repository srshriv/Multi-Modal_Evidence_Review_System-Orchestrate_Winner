"""
Metrics for scoring predictions against the labeled dataset/sample_claims.csv.

Deliberately field-by-field rather than one blended "accuracy" number, to catch
a known failure mode in agent evaluation: a correct label paired with a
justification that is empty, generic, or contradicts the agent's own status.
Field-by-field scoring plus a separate justification-groundedness check surfaces
that decoupling, rather than letting a right label mask a useless justification.
"""

from dataclasses import dataclass, field


EXACT_MATCH_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "valid_image",
    "severity",
]


@dataclass
class FieldScore:
    field: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class RunScore:
    strategy_name: str
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    risk_flag_jaccard_sum: float = 0.0
    risk_flag_rows: int = 0
    justification_nonempty_count: int = 0
    justification_mentions_image_id_count: int = 0
    total_rows: int = 0
    claim_status_confusion: dict = field(default_factory=dict)  # (expected, predicted) -> count
    claim_status_mismatches: list = field(default_factory=list)  # per-row detail, see score_run
    total_api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_images_processed: int = 0
    failures: int = 0
    elapsed_seconds: float = 0.0

    @property
    def overall_field_accuracy(self) -> float:
        if not self.field_scores:
            return 0.0
        return sum(fs.accuracy for fs in self.field_scores.values()) / len(self.field_scores)

    @property
    def risk_flag_avg_jaccard(self) -> float:
        return self.risk_flag_jaccard_sum / self.risk_flag_rows if self.risk_flag_rows else 0.0

    @property
    def justification_groundedness_rate(self) -> float:
        return (
            self.justification_mentions_image_id_count / self.total_rows
            if self.total_rows else 0.0
        )


def _set_from_semicolon(value: str) -> set:
    if not value or value.strip().lower() == "none":
        return set()
    return set(v.strip() for v in value.split(";") if v.strip())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def score_run(strategy_name: str, predictions: list[dict], expected_rows: list[dict]) -> RunScore:
    """
    predictions and expected_rows must be aligned by index (same claim order).
    predictions: list of output row dicts (as produced by pipeline.process_claim)
    expected_rows: list of `claim.expected` dicts from ClaimRow (sample_claims.csv ground truth)
    """
    score = RunScore(strategy_name=strategy_name)
    score.total_rows = len(predictions)

    for f in EXACT_MATCH_FIELDS:
        score.field_scores[f] = FieldScore(field=f)

    for pred, expected in zip(predictions, expected_rows):
        for f in EXACT_MATCH_FIELDS:
            fs = score.field_scores[f]
            fs.total += 1
            if str(pred.get(f, "")).strip().lower() == str(expected.get(f, "")).strip().lower():
                fs.correct += 1

        # risk_flags: set-based similarity rather than exact string match, since
        # flag order / minor flag-set differences shouldn't zero out the score
        # the way a strict string compare would.
        pred_flags = _set_from_semicolon(pred.get("risk_flags", ""))
        exp_flags = _set_from_semicolon(expected.get("risk_flags", ""))
        score.risk_flag_jaccard_sum += _jaccard(pred_flags, exp_flags)
        score.risk_flag_rows += 1

        # claim_status confusion matrix, the single most decision-critical field
        key = (expected.get("claim_status", ""), pred.get("claim_status", ""))
        score.claim_status_confusion[key] = score.claim_status_confusion.get(key, 0) + 1
        if expected.get("claim_status", "") != pred.get("claim_status", ""):
            # Per-row detail for whichever rows disagree, specifically so a
            # confusion-matrix cell like "contradicted -> supported" (the
            # dangerous direction -- a false approval) can be traced back to an
            # actual user_id and investigated, rather than left as an aggregate
            # count with no way to find the row.
            score.claim_status_mismatches.append({
                "user_id": pred.get("user_id", "?"),
                "expected": expected.get("claim_status", ""),
                "predicted": pred.get("claim_status", ""),
                "predicted_justification": pred.get("claim_status_justification", ""),
                "supporting_image_ids": pred.get("supporting_image_ids", ""),
            })

        justification = pred.get("claim_status_justification", "") or ""
        if justification.strip():
            score.justification_nonempty_count += 1
        # Groundedness heuristic: does the justification text reference at least
        # one image_id (either one of the claim's actual supporting IDs, or any
        # img_N pattern), as opposed to a generic sentence with no specific
        # evidence pointer.
        supporting_ids = _set_from_semicolon(pred.get("supporting_image_ids", ""))
        mentions_id = any(iid in justification for iid in supporting_ids) or any(
            f"img_{n}" in justification for n in range(1, 10)
        )
        if mentions_id:
            score.justification_mentions_image_id_count += 1

    return score


def format_score_summary(score: RunScore) -> str:
    lines = [f"### Strategy: {score.strategy_name}", ""]
    lines.append(f"- Rows evaluated: {score.total_rows}")
    lines.append(f"- Failures (fell back to manual-review row): {score.failures}")
    lines.append(f"- Overall field accuracy (avg of 6 exact-match fields): {score.overall_field_accuracy:.1%}")
    lines.append("")
    lines.append("| Field | Accuracy | Correct / Total |")
    lines.append("|---|---|---|")
    for f, fs in score.field_scores.items():
        lines.append(f"| {f} | {fs.accuracy:.1%} | {fs.correct}/{fs.total} |")
    lines.append("")
    lines.append(f"- risk_flags average Jaccard similarity: {score.risk_flag_avg_jaccard:.1%}")
    lines.append(
        f"- Justifications grounded with an image ID reference: "
        f"{score.justification_groundedness_rate:.1%} ({score.justification_mentions_image_id_count}/{score.total_rows})"
    )
    lines.append("")
    lines.append("**claim_status confusion (expected -> predicted):**")
    lines.append("")
    lines.append("| Expected | Predicted | Count |")
    lines.append("|---|---|---|")
    for (exp, pred), count in sorted(score.claim_status_confusion.items()):
        lines.append(f"| {exp} | {pred} | {count} |")
    lines.append("")

    if score.claim_status_mismatches:
        lines.append("**Mismatched rows (for direct debugging):**")
        lines.append("")
        lines.append("| user_id | Expected | Predicted | supporting_image_ids | Justification |")
        lines.append("|---|---|---|---|---|")
        for m in score.claim_status_mismatches:
            just = m["predicted_justification"].replace("|", "/").replace("\n", " ")
            if len(just) > 100:
                just = just[:100] + "..."
            lines.append(
                f"| {m['user_id']} | {m['expected']} | {m['predicted']} | "
                f"{m['supporting_image_ids']} | {just} |"
            )
        lines.append("")
    lines.append(
        f"- API calls: {score.total_api_calls} | input tokens: {score.total_input_tokens} "
        f"| output tokens: {score.total_output_tokens} | elapsed: {score.elapsed_seconds:.1f}s"
    )
    return "\n".join(lines)
