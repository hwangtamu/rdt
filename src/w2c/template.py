"""Canonical prompt template shared by both backbones (plan section 5.1).

Only semantic content goes in: tool definitions that are actually available and
the user request. Answer-bearing metadata (correct_answer, target_tool, gold
assistant text) must never be rendered here.
"""

from __future__ import annotations

import json

DEFAULT_TEMPLATE = (
    "Available Tools:\n{tool_definitions}\n\n"
    "User Request:\n{question}\n\n"
    "Decision:"
)

DEFAULT_TOOL_FORMAT = "- {name}\n  description: {description}\n  parameters: {parameters}"


def _render_parameters(params) -> str:
    if params is None:
        return "(none)"
    if isinstance(params, str):
        return params
    if isinstance(params, dict):
        # Accept {"type": "object", "properties": {...}} JSON-schema style.
        props = params.get("properties", params)
        required = set(params.get("required", []) or [])
        if isinstance(props, dict):
            parts = []
            for name, spec in props.items():
                spec = spec if isinstance(spec, dict) else {}
                typ = spec.get("type", "any")
                req = "required" if name in required else "optional"
                desc = spec.get("description")
                parts.append(f"{name} ({typ}, {req})" + (f": {desc}" if desc else ""))
            return "; ".join(parts) if parts else "(none)"
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def render_tools(tools, tool_format: str = DEFAULT_TOOL_FORMAT,
                 empty_tools: str = "(none)") -> str:
    """Render the available-tool block deterministically, preserving input order."""
    if tools is None:
        return empty_tools
    if isinstance(tools, str):
        return tools.strip() or empty_tools
    if not isinstance(tools, (list, tuple)):
        tools = [tools]

    rendered = []
    for tool in tools:
        if isinstance(tool, str):
            rendered.append(tool)
            continue
        if not isinstance(tool, dict):
            rendered.append(str(tool))
            continue
        rendered.append(tool_format.format(
            name=tool.get("name", "(unnamed)"),
            description=(tool.get("description") or "").strip() or "(none)",
            parameters=_render_parameters(tool.get("parameters", tool.get("args"))),
        ))
    return "\n".join(rendered) if rendered else empty_tools


def build_prompt(question: str, tools, template: str = DEFAULT_TEMPLATE,
                 tool_format: str = DEFAULT_TOOL_FORMAT,
                 empty_tools: str = "(none)") -> str:
    return template.format(
        question=question.strip(),
        tool_definitions=render_tools(tools, tool_format, empty_tools),
    )
