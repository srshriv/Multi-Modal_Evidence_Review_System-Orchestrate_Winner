"""
Anthropic provider for the claim-review agent loop.

Wraps the Claude Messages API tool-calling loop behind the shared provider
interface (see the other files in this package), so core/agent.py can drive any
provider without knowing which one it's talking to. This is what makes the
multi-provider comparison in evaluation/ a fair, apples-to-apples comparison
rather than differently-shaped pipelines.
"""

import os
import anthropic

from core.tools import ALL_TOOL_SCHEMAS


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file (see .env.example)."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _format_tools(self):
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in ALL_TOOL_SCHEMAS
        ]

    def build_image_blocks(self, images: list[tuple[str, str, str]]) -> list[dict]:
        """
        images: list of (image_id, base64_data, media_type)
        Returns content blocks: a text label then the image, repeated per image,
        so the model can clearly tell which image_id corresponds to which image.
        """
        blocks = []
        for image_id, b64_data, media_type in images:
            blocks.append({"type": "text", "text": f"Image {image_id}:"})
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
            })
        return blocks

    def run_turn(self, system_prompt: str, messages: list[dict], max_tokens: int = 2048):
        """
        Sends one request to the model and returns the raw response object.
        `messages` is already in Anthropic's role/content format.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=self._format_tools(),
            messages=messages,
        )
        return response

    def extract_tool_calls(self, response) -> list[dict]:
        """Returns [{id, name, input}, ...] for every tool_use block in the response."""
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name, "input": block.input})
        return calls

    def extract_text(self, response) -> str:
        return "".join(b.text for b in response.content if b.type == "text")

    def assistant_message_from_response(self, response) -> dict:
        """Converts the API response into the dict form needed to append to `messages`."""
        return {"role": "assistant", "content": response.content}

    def tool_result_message(self, tool_call_id: str, result_text: str) -> dict:
        return self.batch_tool_results([(tool_call_id, result_text)])

    def batch_tool_results(self, results: list[tuple[str, str]]) -> list[dict]:
        """
        Builds ONE user message containing tool_result blocks for every
        (tool_call_id, result_text) pair given, wrapped in a list for interface
        consistency with other providers (OpenAI/Groq need multiple separate
        messages; Anthropic needs exactly one). Anthropic requires all
        tool_result blocks answering a single assistant turn's tool_use blocks
        to live in one message -- emitting them as separate messages (even
        one-per-call) is rejected with a 400 error. See the bug-history note
        in core/agent.py for how this was found.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": text}
                for call_id, text in results
            ],
        }]

    def usage_from_response(self, response) -> dict:
        return {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
