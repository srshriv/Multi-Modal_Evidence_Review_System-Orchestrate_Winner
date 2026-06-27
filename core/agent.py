"""
The agentic loop: drives one claim from "images attached" to a validated verdict.

This is the part of the system that should read as a real agent, not a scripted
pipeline -- the model decides whether to call inspect_image, decides when it has
enough to submit_verdict, and the loop itself only enforces a hard safety cap on
iterations (see MAX_ITERATIONS). Everything else is model-driven.

Provider-agnostic by design: works against either AnthropicProvider or
OpenAIProvider because both implement the same small interface (run_turn,
extract_tool_calls, build_image_blocks, etc).
"""

import os
from dataclasses import dataclass, field

from core.data_loader import ClaimRow, DatasetRepository, UserHistory
from core.prompts import SYSTEM_PROMPT, build_claim_context_block
from core.schema import ClaimVerdict, SchemaValidationError
from core.tools import execute_lookup_evidence_requirement

MAX_ITERATIONS = 4  # hard safety cap. Lowered from 6 after measuring that each
                     # additional iteration resends the FULL accumulated message
                     # history (stateless chat-completions API), so cost grows
                     # roughly with iteration^2, not linearly. With
                     # MAX_INSPECT_IMAGE_CALLS=2, a well-behaved run needs at most
                     # ~3 iterations (initial reasoning, up to 2 inspections,
                     # submit_verdict can ride along with the last one); 4 leaves
                     # one turn of slack without re-opening the cost blowup this
                     # was lowered to fix.
MAX_INSPECT_IMAGE_CALLS = 2  # hard cap per claim; see note in run_agent_on_claim for why

# Set AGENT_VERBOSE=1 in the environment to print each turn's actual tool
# calls and reasoning text, not just token counts. Off by default to keep
# normal run output readable; turn on when debugging why a claim landed on
# an unexpected verdict.
VERBOSE = os.environ.get("AGENT_VERBOSE", "") == "1"


@dataclass
class AgentRunResult:
    verdict: ClaimVerdict | None
    raw_verdict_input: dict | None
    iterations_used: int
    tool_calls_made: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    api_calls_made: int = 0
    inspect_image_calls: int = 0
    error: str | None = None


def run_agent_on_claim(
    provider,
    claim: ClaimRow,
    repository: DatasetRepository,
    user_history: UserHistory | None,
) -> AgentRunResult:
    # Load all images for this claim upfront -- the model reasons over the full
    # set from the start, and can request a closer look at any one of them via
    # inspect_image rather than fetching them one at a time blind. See the
    # architecture discussion in evaluation_report.md for why this beats pure
    # sequential tool-fetching for this task.
    images_b64 = []
    for image_id, rel_path in zip(claim.image_ids, claim.image_paths):
        b64_data, media_type = repository.load_image_b64(rel_path)
        images_b64.append((image_id, b64_data, media_type))

    image_blocks = provider.build_image_blocks(images_b64)
    context_text = build_claim_context_block(claim, user_history, repository)

    messages = [
        {
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": context_text}],
        }
    ]

    result = AgentRunResult(verdict=None, raw_verdict_input=None, iterations_used=0)
    schema_retry_used = False
    last_model_text = ""  # tracks the model's most recent free-text reasoning, used to
                           # build an informative (not generic) message if we exhaust
                           # MAX_ITERATIONS without a submit_verdict call

    for iteration in range(1, MAX_ITERATIONS + 1):
        result.iterations_used = iteration
        try:
            response = provider.run_turn(SYSTEM_PROMPT, messages)
        except Exception as e:
            error_text = str(e)
            # Some providers (observed with Groq/Llama 4 Scout) validate tool-call
            # argument types server-side and reject the whole call with a 400 if,
            # e.g., a boolean field was emitted as the string "true" instead of a
            # JSON boolean -- before we ever see a parseable response to retry
            # against. Rather than failing the whole claim on a single formatting
            # slip, give the model exactly one corrective nudge and retry the same
            # turn. This is a narrow, explicit retry path (not blanket retry-on-
            # any-error) so it can't mask genuine reasoning failures.
            is_tool_schema_error = "tool_use_failed" in error_text or (
                "schema" in error_text.lower() and "tool" in error_text.lower()
            )
            if is_tool_schema_error and not schema_retry_used:
                schema_retry_used = True
                # Note: because the call was rejected server-side, we never received
                # a parseable response and therefore cannot append the model's
                # (invalid) assistant turn before this correction. We append the
                # correction as a fresh user turn instead. This works for Groq's
                # chat-completions-style API (no strict alternation requirement),
                # which is the only provider this retry path has been observed to
                # trigger for. If this pattern is ever seen on a strictly-alternating
                # provider, this needs the original raw tool-call payload re-attached
                # alongside the correction instead.
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous tool call was rejected because one or more fields had the "
                        "wrong JSON type. Re-issue the same tool call, but make sure "
                        "evidence_standard_met and valid_image are literal JSON booleans (true / "
                        "false, no quotation marks), and risk_flags / supporting_image_ids are JSON "
                        "arrays of strings, not a single string."
                    ),
                })
                continue
            result.error = f"Provider call failed at iteration {iteration}: {e}"
            return result

        result.api_calls_made += 1
        usage = provider.usage_from_response(response)
        turn_input_tokens = usage.get("input_tokens", 0)
        turn_output_tokens = usage.get("output_tokens", 0)
        result.total_input_tokens += turn_input_tokens
        result.total_output_tokens += turn_output_tokens
        # Each turn resends the full message history (system prompt + every prior
        # image and tool result, since chat-completions-style APIs are stateless),
        # so input tokens grow with iteration count, not just claim complexity.
        # This print is intentionally left in (not behind a verbose flag) because
        # diagnosing exactly this growth pattern is what's needed when a run hits
        # a provider's TPM/TPD limit -- see evaluation_report.md's operational
        # analysis section for how these numbers get used.
        print(f"        iter {iteration}: in={turn_input_tokens} out={turn_output_tokens} "
              f"(running total in={result.total_input_tokens})")

        messages.append(provider.assistant_message_from_response(response))
        tool_calls = provider.extract_tool_calls(response)
        turn_text = provider.extract_text(response)
        if turn_text.strip():
            last_model_text = turn_text.strip()

        if VERBOSE:
            # Surfaces what the model actually did each turn -- which tools it
            # called with what arguments, and any free-text reasoning -- rather
            # than only the token-count summary. Added after a real debugging
            # session where a run kept landing on not_enough_information
            # repeatedly and there was no way to see why from the default logs.
            if turn_text.strip():
                print(f"        [reasoning] {turn_text.strip()[:500]}")
            for call in tool_calls:
                print(f"        [tool_call] {call['name']}({call['input']})")

        if not tool_calls:
            # Model responded with plain text and no tool call. Nudge it back
            # toward using submit_verdict rather than silently failing the claim.
            messages.append({
                "role": "user",
                "content": "Please call a tool (inspect_image, lookup_evidence_requirement, "
                           "or submit_verdict) to continue the review.",
            })
            continue

        verdict_call = next((c for c in tool_calls if c["name"] == "submit_verdict"), None)

        # Budget-aware closure: once we're within striking distance of the hard
        # cap and the model still hasn't submitted a verdict, stop processing
        # any other tool call and force it toward submit_verdict instead.
        #
        # BUG HISTORY (kept here because it cost real, scarce eval-run quota to
        # find): the first version of this only checked
        # `iteration == MAX_ITERATIONS - 1`, fired the nudge ONCE, then
        # `continue`d straight to the final iteration with NO further
        # enforcement -- if the model ignored the nudge and called another tool
        # anyway on the last turn, that call was processed normally (since the
        # condition was false again) and the loop then exhausted with a generic
        # failure, discarding three turns of real reasoning. Fixed by checking
        # `iteration >= MAX_ITERATIONS - 1` (an inequality, not equality) so the
        # block applies on EVERY remaining turn once triggered, not just one.
        if verdict_call is None and iteration >= MAX_ITERATIONS - 1:
            # BUG FIX: previously this appended one separate user-role message
            # per tool call via provider.tool_result_message(). Anthropic's API
            # requires ALL tool_result blocks for a single assistant turn to be
            # batched together in ONE user message's content array -- two
            # consecutive single-result user messages are invalid even though
            # each one looks correct in isolation, and Anthropic rejects the
            # whole request with "tool_use ids were found without tool_result
            # blocks immediately after". This only ever showed up on turns
            # where the model made 2+ tool calls at once, which Groq's smaller
            # model rarely did but Claude does fairly often. Fixed by using
            # provider.batch_tool_results() to build one message instead.
            results = [
                (call["id"], "Tool budget exhausted for this claim. You MUST call submit_verdict on "
                 "your very next turn -- no other tool calls will be processed. Decide now based on "
                 "everything you have already observed. A confident not_enough_information is "
                 "a complete, valid answer if you genuinely cannot determine more; it is far "
                 "better than not answering at all.")
                for call in tool_calls
            ]
            messages.extend(provider.batch_tool_results(results))
            continue

        if verdict_call:
            result.tool_calls_made.append("submit_verdict")
            try:
                result.raw_verdict_input = verdict_call["input"]
                result.verdict = ClaimVerdict(
                    evidence_standard_met=verdict_call["input"]["evidence_standard_met"],
                    evidence_standard_met_reason=verdict_call["input"]["evidence_standard_met_reason"],
                    risk_flags=verdict_call["input"].get("risk_flags", []),
                    issue_type=verdict_call["input"]["issue_type"],
                    object_part=verdict_call["input"]["object_part"],
                    claim_status=verdict_call["input"]["claim_status"],
                    claim_status_justification=verdict_call["input"]["claim_status_justification"],
                    supporting_image_ids=verdict_call["input"].get("supporting_image_ids", []),
                    valid_image=verdict_call["input"]["valid_image"],
                    severity=verdict_call["input"]["severity"],
                )
            except (KeyError, SchemaValidationError) as e:
                result.error = f"Malformed submit_verdict payload: {e}"
            return result

        # Handle inspect_image / lookup_evidence_requirement calls, then loop again.
        # All tool_result blocks for this turn must be batched into a single
        # message (see the BUG FIX note above) -- collect them first, emit one
        # batched message, then attach any re-inspected images as a separate
        # follow-up user turn.
        pending_results = []
        pending_image_reattachments = []
        for call in tool_calls:
            result.tool_calls_made.append(call["name"])
            if call["name"] == "inspect_image":
                if result.inspect_image_calls >= MAX_INSPECT_IMAGE_CALLS:
                    # Hard budget: each inspect_image call re-attaches a full image
                    # to the growing message history, and that history is resent in
                    # full on every subsequent turn. Unbounded re-inspection was the
                    # main driver behind hitting Groq's per-day token limit in
                    # practice (see evaluation_report.md). Two closer looks per claim
                    # is enough for genuine uncertainty; beyond that we redirect the
                    # model to decide with what it has rather than let cost balloon.
                    tool_text = (
                        f"inspect_image budget exhausted ({MAX_INSPECT_IMAGE_CALLS} used for this "
                        "claim). Decide based on what you've already observed and call submit_verdict "
                        "-- if you genuinely cannot determine the answer, that itself is a valid "
                        "not_enough_information verdict."
                    )
                    pending_results.append((call["id"], tool_text))
                    continue
                result.inspect_image_calls += 1
                image_id = call["input"].get("image_id", "")
                focus = call["input"].get("focus_question", "")
                match = next((img for img in images_b64 if img[0] == image_id), None)
                if match is None:
                    tool_text = f"No image with id '{image_id}' was found in this claim's image set."
                    pending_results.append((call["id"], tool_text))
                else:
                    # Re-attach the image itself as a follow-up so the provider
                    # actually gets a closer "look" rather than a text-only reply.
                    tool_text = f"Re-attached for closer inspection. Focus question: {focus}"
                    pending_results.append((call["id"], tool_text))
                    pending_image_reattachments.append(match)
            elif call["name"] == "lookup_evidence_requirement":
                query = call["input"].get("issue_family_query", "")
                tool_text = execute_lookup_evidence_requirement(repository, claim.claim_object, query)
                pending_results.append((call["id"], tool_text))
            else:
                pending_results.append((call["id"], "Unknown tool."))

        messages.extend(provider.batch_tool_results(pending_results))
        if pending_image_reattachments:
            messages.append({
                "role": "user",
                "content": provider.build_image_blocks(pending_image_reattachments),
            })

    # True backstop: the in-loop closure nudge (above) asks nicely but cannot
    # force compliance -- a model can still ignore "call submit_verdict" and
    # call another tool instead, right up to the last budgeted iteration. If
    # that happens, make exactly ONE more dedicated call, outside the normal
    # MAX_ITERATIONS budget, with a message that makes submit_verdict the only
    # sane response and nothing else. This costs one extra call in the rare
    # case it's needed, which is cheap insurance against losing a claim's
    # entire reasoning history (and the tokens already spent producing it) to
    # a fallback row with no real verdict.
    messages.append({
        "role": "user",
        "content": (
            "You are out of tool-call budget for this claim. This is your final turn. "
            "Call submit_verdict now with your best assessment based on everything observed "
            "so far in this conversation. If you are genuinely uncertain, submit "
            "claim_status='not_enough_information' with an honest justification -- but you "
            "must call submit_verdict; no other tool will be processed."
        ),
    })
    try:
        response = provider.run_turn(SYSTEM_PROMPT, messages)
        result.api_calls_made += 1
        usage = provider.usage_from_response(response)
        result.total_input_tokens += usage.get("input_tokens", 0)
        result.total_output_tokens += usage.get("output_tokens", 0)
        tool_calls = provider.extract_tool_calls(response)
        verdict_call = next((c for c in tool_calls if c["name"] == "submit_verdict"), None)
        if verdict_call:
            result.tool_calls_made.append("submit_verdict")
            try:
                result.raw_verdict_input = verdict_call["input"]
                result.verdict = ClaimVerdict(
                    evidence_standard_met=verdict_call["input"]["evidence_standard_met"],
                    evidence_standard_met_reason=verdict_call["input"]["evidence_standard_met_reason"],
                    risk_flags=verdict_call["input"].get("risk_flags", []),
                    issue_type=verdict_call["input"]["issue_type"],
                    object_part=verdict_call["input"]["object_part"],
                    claim_status=verdict_call["input"]["claim_status"],
                    claim_status_justification=verdict_call["input"]["claim_status_justification"],
                    supporting_image_ids=verdict_call["input"].get("supporting_image_ids", []),
                    valid_image=verdict_call["input"]["valid_image"],
                    severity=verdict_call["input"]["severity"],
                )
                return result
            except (KeyError, SchemaValidationError) as e:
                result.error = f"Malformed submit_verdict payload on backstop turn: {e}"
                return result
    except Exception as e:
        result.error = f"Backstop closure call also failed: {e}"
        return result

    if last_model_text:
        result.error = (
            f"Exceeded max iterations ({MAX_ITERATIONS}) and backstop closure turn without "
            f"submit_verdict. Model's last reasoning before cutoff: {last_model_text[:500]}"
        )
    else:
        result.error = (
            f"Exceeded max iterations ({MAX_ITERATIONS}) and backstop closure turn without "
            "submit_verdict."
        )
    return result
