"""
OpenAI provider for the claim-review agent loop.

Same responsibilities as AnthropicProvider, adapted to the Chat Completions
tool-calling format. Kept structurally parallel on purpose -- same method
names, same return shapes from agent.py's point of view -- so swapping
providers is a one-line change in main.py / evaluation/main.py.
"""

import json
import os
from openai import OpenAI

from core.tools import ALL_TOOL_SCHEMAS


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file (see .env.example)."
            )
        self.client = OpenAI(api_key=api_key)
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
        """
        images: list of (image_id, base64_data, media_type)
        OpenAI's chat content blocks: a text label then an image_url data-uri,
        repeated per image, mirroring the Anthropic provider's labeling scheme.
        """
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
        """
        OpenAI's chat-completions format wants one separate {"role": "tool", ...}
        message per tool call (not batched into a single message's content
        array, unlike Anthropic) -- see core/providers/anthropic_provider.py for
        why this method exists at all. Returns a list so agent.py can extend()
        messages with it uniformly across providers regardless of whether the
        underlying format wants one message or several.
        """
        return [self.tool_result_message(call_id, text) for call_id, text in results]

    def usage_from_response(self, response) -> dict:
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
