"""
Deterministic data loading for the claim review pipeline.

Nothing in this file calls a model. It is plain CSV/filesystem plumbing:
reading claims, resolving image paths relative to the dataset root, looking up
user history, and matching evidence-requirement rows. Keeping this deterministic
and separate from the agent loop means these lookups are reproducible, free,
and don't burn a model call on work a dict lookup already does perfectly.
"""

import base64
import csv
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Vision-model token cost scales with image resolution, not with how much
# damage-relevant detail is actually in the photo. The sample dataset includes
# images up to ~5.7MB at full camera resolution, which is far more pixels than
# a claim-review judgment (is there a dent, is there a watermark, is there a
# sticky note) needs, and burns a disproportionate number of input tokens per
# claim. We resize to a bounded max dimension and re-encode as JPEG before
# sending to any provider. This was added after a live run against Groq
# measured ~12,400 input tokens/claim and hit the daily token cap mid-run;
# resizing is the direct fix, not just a nice-to-have.
MAX_IMAGE_DIMENSION = 896  # longest side, in pixels. Lowered from 1024 after a
# real Groq evaluation run measured 7,000-15,000 input tokens/claim and still
# came close to the 500K tokens/day free-tier cap; 896px was visually verified
# (not assumed) to still preserve both the case_008 stock-photo watermark and
# the case_020 prompt-injection sticky-note text, while cutting ~20% more
# tokens off image-heavy claims versus 1024px. Going lower (768px) was tested
# but not adopted without further visual verification across more of the
# adversarial sample cases -- 896px was chosen as the point with confirmed
# legibility on the two cases that matter most for safety-gate behavior.
JPEG_QUALITY = 85


@dataclass
class ClaimRow:
    user_id: str
    image_paths: list[str]          # raw relative paths, e.g. images/test/case_001/img_1.jpg
    user_claim: str
    claim_object: str
    # ground truth fields, only present when loading sample_claims.csv
    expected: dict | None = None

    @property
    def image_ids(self) -> list[str]:
        """Filename without extension, e.g. 'img_1'. This is the ID used in supporting_image_ids."""
        return [Path(p).stem for p in self.image_paths]


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

    @property
    def is_high_risk(self) -> bool:
        """
        Deterministic risk threshold, separate from anything the model decides.
        A user is treated as high-risk history if the dataset already flags them,
        or if their rejection rate is high enough to be a meaningful signal.
        This logic is intentionally simple and auditable -- it is a guardrail
        input to the agent, not a replacement for looking at the images.
        """
        if self.history_flags and self.history_flags.strip().lower() not in ("none", ""):
            return True
        if self.past_claim_count >= 3 and self.rejected_claim / max(self.past_claim_count, 1) >= 0.3:
            return True
        return False


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_object: str   # 'car' | 'laptop' | 'package' | 'all'
    applies_to: str
    minimum_image_evidence: str


class DatasetRepository:
    """Loads and serves the four CSV inputs plus image resolution, relative to a dataset root."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self._user_history: dict[str, UserHistory] = {}
        self._evidence_requirements: list[EvidenceRequirement] = []
        self._load_user_history()
        self._load_evidence_requirements()

    def _load_user_history(self):
        path = self.dataset_root / "user_history.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uh = UserHistory(
                    user_id=row["user_id"],
                    past_claim_count=int(row["past_claim_count"]),
                    accept_claim=int(row["accept_claim"]),
                    manual_review_claim=int(row["manual_review_claim"]),
                    rejected_claim=int(row["rejected_claim"]),
                    last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                    history_flags=row["history_flags"],
                    history_summary=row["history_summary"],
                )
                self._user_history[uh.user_id] = uh

    def _load_evidence_requirements(self):
        path = self.dataset_root / "evidence_requirements.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._evidence_requirements.append(
                    EvidenceRequirement(
                        requirement_id=row["requirement_id"],
                        claim_object=row["claim_object"],
                        applies_to=row["applies_to"],
                        minimum_image_evidence=row["minimum_image_evidence"],
                    )
                )

    def get_user_history(self, user_id: str) -> UserHistory | None:
        return self._user_history.get(user_id)

    def get_evidence_requirements(self, claim_object: str) -> list[EvidenceRequirement]:
        """All requirement rows relevant to this object: object-specific + 'all'."""
        return [
            r for r in self._evidence_requirements
            if r.claim_object == claim_object or r.claim_object == "all"
        ]

    def load_claims_csv(self, filename: str, has_labels: bool = False) -> list[ClaimRow]:
        path = self.dataset_root / filename
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                expected = None
                if has_labels:
                    expected = {
                        "evidence_standard_met": row.get("evidence_standard_met"),
                        "evidence_standard_met_reason": row.get("evidence_standard_met_reason"),
                        "risk_flags": row.get("risk_flags"),
                        "issue_type": row.get("issue_type"),
                        "object_part": row.get("object_part"),
                        "claim_status": row.get("claim_status"),
                        "claim_status_justification": row.get("claim_status_justification"),
                        "supporting_image_ids": row.get("supporting_image_ids"),
                        "valid_image": row.get("valid_image"),
                        "severity": row.get("severity"),
                    }
                rows.append(
                    ClaimRow(
                        user_id=row["user_id"],
                        image_paths=[p.strip() for p in row["image_paths"].split(";") if p.strip()],
                        user_claim=row["user_claim"],
                        claim_object=row["claim_object"],
                        expected=expected,
                    )
                )
        return rows

    def resolve_image_path(self, relative_path: str) -> Path:
        return self.dataset_root / relative_path

    def load_image_b64(self, relative_path: str) -> tuple[str, str]:
        """
        Returns (base64_data, media_type) for a given dataset-relative image path,
        after resizing to MAX_IMAGE_DIMENSION and re-encoding as JPEG. This keeps
        vision-model token cost per image bounded and predictable regardless of
        the original photo's resolution (see MAX_IMAGE_DIMENSION comment above).

        Important: resizing must not destroy the signal we actually need to
        detect -- stock-photo watermarks, small in-image text, fine scratches.
        1024px on the longest side comfortably preserves watermark text and
        sticky-note handwriting (verified visually during development against
        dataset/images/sample/case_008 and case_020), while cutting token cost
        roughly in proportion to the area reduction versus the original.
        """
        full_path = self.resolve_image_path(relative_path)
        with Image.open(full_path) as img:
            img = img.convert("RGB")  # normalize mode (handles PNG-with-alpha, CMYK, etc.)
            width, height = img.size
            longest_side = max(width, height)
            if longest_side > MAX_IMAGE_DIMENSION:
                scale = MAX_IMAGE_DIMENSION / longest_side
                new_size = (round(width * scale), round(height * scale))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
            data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        return data, "image/jpeg"


def write_output_csv(rows: list[dict], output_path: Path, columns: list[str]):
    """Writes final predictions, columns in exact required order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
