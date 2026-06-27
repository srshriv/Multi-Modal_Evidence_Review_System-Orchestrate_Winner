"""
Google Gemini provider for the claim-review agent loop.

Added as a fourth strategy after Anthropic credit, OpenAI quota, and Groq's
500K-tokens/day free-tier cap all became practical blockers during
development -- Gemini's free tier (via Google AI Studio, no credit card) is
dramatically more generous: 1,500 requests/day and 1,000,000 tokens/minute on
Gemini 2.5 Flash as of mid-2026, versus Groq's free tier's much tighter daily
token ceiling. This was a real, measured constraint hit during the build, not
a hypothetical -- see evaluation_report.md and the chat transcript for the
debugging history.

Uses the current `google-genai` SDK (`pip install google-genai`,
`from google import genai`), NOT the deprecated `google-generativeai` package
(deprecated August 2025, no longer recommended by Google as of this build).

Structurally this provider implements the same small interface as
AnthropicProvider / OpenAIProvider / GroqProvider (run_turn,
extract_tool_calls, build_image_blocks, etc.) so core/agent.py can drive it
without any provider-specific branching. The underlying request/response shape
is different (Gemini uses `contents` made of `Part` objects and
`FunctionDeclaration`-based tools rather than `messages` and OpenAI-style
`tools`), so this adapter does real translation work, not a thin URL swap
like GroqProvider was able to get away with.
"""

import os

from google import genai
from google.genai import types

from core.tools import ALL_TOOL_SCHEMAS


def _json_schema_to_gemini_schema(schema: dict) -> types.Schema:
    """
    Translates our provider-agnostic JSON Schema (used for Anthropic/OpenAI/Groq
    tool definitions in core/tools.py and core/schema.py) into a
    google.genai.types.Schema object, which Gemini's FunctionDeclaration
    requires instead of raw JSON Schema dicts.
    """
    type_map = {
        "object": types.Type.OBJECT,
        "string": types.Type.STRING,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
    }

    json_type = schema.get("type", "string")
    gemini_type = type_map.get(json_type, types.Type.STRING)

    kwargs = {"type": gemini_type}
    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "enum" in schema:
        kwargs["enum"] = schema["enum"]

    if json_type == "object" and "properties" in schema:
        kwargs["properties"] = {
            k: _json_schema_to_gemini_schema(v) for k, v in schema["properties"].items()
        }
        if "required" in schema:
            kwargs["required"] = schema["required"]

    if json_type == "array" and "items" in schema:
        kwargs["items"] = _json_schema_to_gemini_schema(schema["items"])

    return types.Schema(**kwargs)


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env file (see .env.example). "
                "Get a free key with no credit card at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _format_tools(self) -> list[types.Tool]:
        declarations = []
        for t in ALL_TOOL_SCHEMAS:
            declarations.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_json_schema_to_gemini_schema(t["input_schema"]),
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def build_image_blocks(self, images: list[tuple[str, str, str]]) -> list:
        """
        images: list of (image_id, base64_data, media_type)
        Returns a list of content parts (label text + inline image bytes) in
        Gemini's Part format, mirroring the label-then-image pattern used by
        the other providers so the model can tell which image_id is which.
        """
        import base64
        parts = []
        for image_id, b64_data, media_type in images:
            parts.append(types.Part.from_text(text=f"Image {image_id}:"))
            parts.append(
                types.Part.from_bytes(data=base64.standard_b64decode(b64_data), mime_type=media_type)
            )
        return parts

    def run_turn(self, system_prompt: str, messages: list[dict], max_tokens: int = 2048):
        """
        `messages` arrives in the provider-agnostic role/content shape used by
        agent.py. We translate it into Gemini's `contents` list of
        types.Content objects on every call, since Gemini's SDK is stateless
        per-call just like the others (full history resent each turn).
        """
        contents = self._messages_to_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=self._format_tools(),
            max_output_tokens=max_tokens,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        return response

    def _messages_to_contents(self, messages: list[dict]) -> list[types.Content]:
        """
        Converts our internal message list (built up by agent.py using
        assistant_message_from_response / tool_result_message / plain dicts
        with role+content) into Gemini Content objects. Each internal message
        is one of:
          - {"role": "user"/"assistant", "content": [parts...] or "text"}
          - the dict returned by assistant_message_from_response (below)
          - the dict returned by tool_result_message (below)
        """
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]

            if isinstance(content, str):
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=content)]))
            elif isinstance(content, list):
                # Already a list of types.Part (from build_image_blocks) or our
                # internal function-call/function-response marker dicts.
                parts = []
                for item in content:
                    if isinstance(item, types.Part):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("_gemini_function_call"):
                        parts.append(types.Part.from_function_call(
                            name=item["name"], args=item["args"],
                        ))
                    elif isinstance(item, dict) and item.get("_gemini_function_response"):
                        parts.append(types.Part.from_function_response(
                            name=item["name"], response={"result": item["result"]},
                        ))
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(types.Part.from_text(text=item["text"]))
                contents.append(types.Content(role=role, parts=parts))
        return contents

    def extract_tool_calls(self, response) -> list[dict]:
        calls = []
        if not response.candidates:
            return calls
        for part in response.candidates[0].content.parts:
            if part.function_call:
                calls.append({
                    "id": part.function_call.name,  # Gemini has no call-id concept; name stands in
                    "name": part.function_call.name,
                    "input": dict(part.function_call.args) if part.function_call.args else {},
                })
        return calls

    def extract_text(self, response) -> str:
        if not response.candidates:
            return ""
        texts = [p.text for p in response.candidates[0].content.parts if p.text]
        return "".join(texts)

    def assistant_message_from_response(self, response) -> dict:
        """
        Builds the internal-format assistant turn. Function calls are encoded
        as marker dicts (_gemini_function_call) so _messages_to_contents can
        reconstruct proper Gemini Part.from_function_call objects later --
        Gemini requires the function-call part to be replayed back to it
        exactly when we send the function response on the next turn.
        """
        content_items = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    content_items.append({
                        "_gemini_function_call": True,
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {},
                    })
                elif part.text:
                    content_items.append({"text": part.text})
        return {"role": "assistant", "content": content_items}

    def tool_result_message(self, tool_call_id: str, result_text: str) -> dict:
        # tool_call_id here is actually the function name (see extract_tool_calls
        # note above) since Gemini has no separate call-id concept.
        return self.batch_tool_results([(tool_call_id, result_text)])[0]

    def batch_tool_results(self, results: list[tuple[str, str]]) -> list[dict]:
        """
        Gemini, like Anthropic, expects all function-response parts for a
        single turn bundled into one Content message rather than emitted as
        separate messages -- see AnthropicProvider.batch_tool_results for the
        underlying bug this was added to fix. Returned as a one-item list for
        interface consistency with providers that need multiple messages.
        """
        return [{
            "role": "user",
            "content": [
                {"_gemini_function_response": True, "name": call_id, "result": text}
                for call_id, text in results
            ],
        }]

    def usage_from_response(self, response) -> dict:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0}
        return {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        }
