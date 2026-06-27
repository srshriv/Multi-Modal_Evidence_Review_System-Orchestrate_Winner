"""
Groq provider for the claim-review agent loop.

Groq exposes an OpenAI-compatible chat.completions API, so rather than adding
a new SDK dependency we reuse the `openai` Python client pointed at Groq's
base URL (this is Groq's own documented integration path). Structurally this
provider is almost identical to OpenAIProvider -- same method names, same
tool-calling shape -- which is what lets core/agent.py drive it without any
special-casing.

Model: meta-llama/llama-4-scout-17b-16e-instruct
  - Chosen specifically because it is one of the few models still live on Groq
    that is actually vision-capable. Groq deprecated Llama 4 Maverick on
    March 9, 2026 in favor of openai/gpt-oss-120b, but gpt-oss-120b is
    text-only -- not a fit for this task, which requires image reasoning.
    Scout remains GA on Groq with native multimodal support, function
    calling/tool use, and JSON mode, so it's the correct choice here, not
    just the only one available.
  - Groq's documented vision limits: up to 5 images per request. Our claims
    data tops out well under that (max image count observed in dataset/claims.csv
    is 3), so no chunking logic is needed for this dataset, but see the note
    in run_turn() if that assumption ever changes.
"""

import json
import os
from openai import OpenAI

from core.tools import ALL_TOOL_SCHEMAS

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MAX_IMAGES_PER_REQUEST = 5  # documented Groq vision limit, see module docstring


class GroqProvider:
    name = "groq"

    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file (see .env.example)."
            )
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.model = model

    def _format_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in ALL_TOOL_SCHEMAS
        ]

    def build_image_blocks(self, images: list[tuple[str, str, str]]) -> list[dict]:
        if len(images) > GROQ_MAX_IMAGES_PER_REQUEST:
            # Defensive guard, not expected to trigger on this dataset (see
            # module docstring) -- if it ever does, this needs a real chunking
            # strategy (e.g. multiple inspect_image follow-ups) rather than
            # silently dropping images.
            raise ValueError(
                f"Groq vision supports at most {GROQ_MAX_IMAGES_PER_REQUEST} images per request, "
                f"got {len(images)}. This claim needs a chunking strategy not yet implemented."
            )
        blocks = []
        for image_id, b64_data, media_type in images:
            blocks.append({"type": "text", "text": f"Image {image_id}:"})
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
            })
        return blocks

    def run_turn(self, system_prompt: str, messages: list[dict], max_tokens: int = 2048):
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=self._format_tools(),
            messages=full_messages,
        )
        return response

    def extract_tool_calls(self, response) -> list[dict]:
        msg = response.choices[0].message
        calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    parsed_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    parsed_input = {}
                calls.append({"id": tc.id, "name": tc.function.name, "input": parsed_input})
        return calls

    def extract_text(self, response) -> str:
        return response.choices[0].message.content or ""

    def assistant_message_from_response(self, response) -> dict:
        msg = response.choices[0].message
        result = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return result

    def tool_result_message(self, tool_call_id: str, result_text: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result_text}

    def batch_tool_results(self, results: list[tuple[str, str]]) -> list[dict]:
        """See OpenAIProvider.batch_tool_results -- Groq uses the same chat-completions
        format (one separate tool-role message per call), since GroqProvider reuses
        the openai SDK pointed at Groq's endpoint."""
        return [self.tool_result_message(call_id, text) for call_id, text in results]

    def usage_from_response(self, response) -> dict:
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
