"""
Tool definitions for the claim-review agent loop.

Three tools, each with a narrow, single responsibility (per the May findings:
"the best multi-agent submissions used those stages... each agent had a narrow
job" -- the same discipline applies to tools within a single agent):

  - inspect_image: re-examine one already-attached image more closely, with a
    specific focus question. This is what makes the loop genuinely agentic --
    the model decides if/when it needs a second look, rather than us scripting
    a fixed sequence of image reads.
  - lookup_evidence_requirement: deterministic dict lookup exposed as a tool so
    the model can explicitly check "do I have what's required" instead of
    guessing from memory.
  - submit_verdict: the forced structured final answer. Defined in schema.py
    and re-exported here so agent.py has one place to import all three from.
"""

from core.schema import SUBMIT_VERDICT_TOOL_SCHEMA


INSPECT_IMAGE_TOOL_SCHEMA = {
    "name": "inspect_image",
    "description": (
        "Take a closer, focused look at one image you have already been shown, "
        "answering a specific question about it (e.g. 'is there a watermark or "
        "stock-photo marking?', 'does this clearly show the rear bumper or a "
        "different part?', 'is there any text/sticky-note/handwriting visible in "
        "this image?'). Use this when your initial read of an image was uncertain "
        "and a targeted re-check would change your decision. Do not use this for "
        "images you have already assessed with confidence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string",
                "description": "The image ID to re-inspect, e.g. 'img_1'.",
            },
            "focus_question": {
                "type": "string",
                "description": "The specific question you want answered about this image.",
            },
        },
        "required": ["image_id", "focus_question"],
    },
}

LOOKUP_EVIDENCE_REQUIREMENT_TOOL_SCHEMA = {
    "name": "lookup_evidence_requirement",
    "description": (
        "Look up the minimum image evidence required for this claim_object and "
        "issue family, from the evidence_requirements reference table. Use this "
        "before deciding evidence_standard_met, so the bar you're applying is the "
        "documented one rather than an assumption."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_family_query": {
                "type": "string",
                "description": (
                    "A short phrase describing the issue family to match against "
                    "the requirements table, e.g. 'dent or scratch', 'crack, broken, "
                    "or missing part', 'crushed, torn, or seal damage'."
                ),
            },
        },
        "required": ["issue_family_query"],
    },
}

ALL_TOOL_SCHEMAS = [
    INSPECT_IMAGE_TOOL_SCHEMA,
    LOOKUP_EVIDENCE_REQUIREMENT_TOOL_SCHEMA,
    SUBMIT_VERDICT_TOOL_SCHEMA,
]


def execute_lookup_evidence_requirement(repository, claim_object: str, issue_family_query: str) -> str:
    """
    Deterministic, no model call. Simple keyword overlap match against the
    evidence_requirements rows for this claim_object (+ 'all' rows).
    Returns a formatted string describing the matched requirement(s).
    """
    requirements = repository.get_evidence_requirements(claim_object)
    query_words = set(issue_family_query.lower().replace(",", " ").split())

    scored = []
    for r in requirements:
        applies_words = set(r.applies_to.lower().replace(",", " ").split())
        overlap = len(query_words & applies_words)
        scored.append((overlap, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for score, r in scored if score > 0][:2]
    if not top:
        # Fall back to the general requirements that apply to every claim_object
        top = [r for r in requirements if r.claim_object == "all"][:2]

    if not top:
        return "No specific requirement matched; apply general judgment about whether the claimed part is visible."

    lines = []
    for r in top:
        lines.append(f"[{r.requirement_id}] (applies_to: {r.applies_to}): {r.minimum_image_evidence}")
    return "\n".join(lines)
