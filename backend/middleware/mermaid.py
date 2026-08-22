from __future__ import annotations
import logging
import re
from typing import Any
from langchain_core.messages import AIMessage
from langchain.agents.middleware.types import AgentMiddleware, AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid first-token diagram types supported by Mermaid 10+
_VALID_DIAGRAM_TYPES = {
    "graph",
    "flowchart",
    "sequencediagram",
    "classDiagram",
    "statediagram",
    "statediagram-v2",
    "erdiagram",
    "journey",
    "gantt",
    "pie",
    "quadrantchart",
    "requirementdiagram",
    "gitgraph",
    "mindmap",
    "timeline",
    "xychart-beta",
    "block-beta",
    "packet-beta",
    "kanban",
    "architecture-beta",
}

# Mermaid reserved words that must NOT be used as bare node IDs
_RESERVED_WORDS = {
    "graph",
    "subgraph",
    "end",
    "style",
    "classDef",
    "click",
    "call",
    "href",
    "linkStyle",
    "class",
    "direction",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _extract_mermaid_blocks(text: str) -> list[tuple[int, int, str]]:
    """
    Find all ```mermaid ... ``` blocks in *text*.

    Returns a list of (start_index, end_index, block_content) tuples so the
    caller can replace them by position without regex on the full content.
    """
    results = []
    for m in re.finditer(
        r"```mermaid\s+(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    ):
        results.append((m.start(), m.end(), m.group(1)))
    return results


def _validate_mermaid_block(code: str) -> list[str]:
    """
    Perform lightweight server-side Mermaid syntax validation.

    Returns a list of human-readable error strings. An empty list means
    the block passed all checks.
    """
    errors: list[str] = []
    lines = [ln.strip() for ln in code.strip().splitlines() if ln.strip()]

    if not lines:
        errors.append("Mermaid block is empty.")
        return errors

    # -----------------------------------------------------------------
    # 1. Check that the first token is a recognised diagram type
    # -----------------------------------------------------------------
    first_line = lines[0]
    first_token = first_line.split()[0].lower().rstrip(":")
    if first_token not in {d.lower() for d in _VALID_DIAGRAM_TYPES}:
        errors.append(
            f"Unknown or missing diagram type on the first line: '{first_line}'. "
            f"Expected one of: flowchart, sequenceDiagram, classDiagram, etc."
        )

    # -----------------------------------------------------------------
    # 2. Detect reserved words used as bare node IDs
    #    Pattern: bare word immediately followed by [ or { or ( or >
    # -----------------------------------------------------------------
    reserved_node_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in _RESERVED_WORDS) + r")\s*[\[\({>]",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines[1:], start=2):
        match = reserved_node_pattern.search(line)
        if match:
            errors.append(
                f"Line {i}: Reserved word '{match.group(1)}' used as a node ID. "
                "Use a different alphanumeric identifier instead."
            )

    # -----------------------------------------------------------------
    # 3. Check for unclosed brackets / parentheses / braces
    # -----------------------------------------------------------------
    bracket_map = {"[": "]", "(": ")", "{": "}"}
    stack: list[tuple[str, int]] = []
    for i, line in enumerate(lines, start=1):
        # Skip comment lines
        if line.startswith("%%"):
            continue
        for ch in line:
            if ch in bracket_map:
                stack.append((ch, i))
            elif ch in bracket_map.values():
                if stack and bracket_map[stack[-1][0]] == ch:
                    stack.pop()
                # Mismatched close bracket — skip to avoid false positives
    if stack:
        locs = ", ".join(f"line {ln}" for _, ln in stack)
        errors.append(f"Unclosed bracket(s) found at: {locs}.")

    # -----------------------------------------------------------------
    # 4. Spaces / hyphens / special chars in node IDs (flowchart/graph)
    # -----------------------------------------------------------------
    if first_token in {"graph", "flowchart"}:
        bad_id_pattern = re.compile(
            r"\b([A-Za-z0-9_]+(?:[\s\-][A-Za-z0-9_]+)+)\s*[\[\({>]"
        )
        for i, line in enumerate(lines[1:], start=2):
            match = bad_id_pattern.search(line)
            if match:
                errors.append(
                    f"Line {i}: Node ID '{match.group(1)}' contains spaces or hyphens. "
                    "Node IDs must be a single alphanumeric word (e.g. 'MyNode')."
                )

    # -----------------------------------------------------------------
    # 5. Check that descriptive node labels are wrapped in double-quotes
    #    e.g.  NodeA[This is bad]  vs  NodeA["This is good"]
    # -----------------------------------------------------------------
    if first_token in {"graph", "flowchart"}:
        unquoted_label_pattern = re.compile(
            r'\[(?!")([^\]]*\s[^\]]*)\]'  # [text with spaces] but NOT ["..."]
        )
        for i, line in enumerate(lines[1:], start=2):
            if line.startswith("%%"):
                continue
            match = unquoted_label_pattern.search(line)
            if match:
                errors.append(
                    f"Line {i}: Node label '{match.group(1)}' contains spaces but is "
                    "not quoted. Wrap multi-word labels in double quotes: "
                    f'["{match.group(1)}"].'
                )

    return errors


# ---------------------------------------------------------------------------
# Auto-fix helpers  (deterministic, no model call)
# ---------------------------------------------------------------------------
def _fix_unquoted_labels(code: str) -> str:
    """
    Wrap unquoted multi-word node labels in double-quotes.
    e.g.  NodeA[My Label]  ->  NodeA["My Label"]
          NodeA(My Label)  ->  NodeA("My Label")
    Leaves already-quoted strings alone.
    """

    def _quote_bracket(m: re.Match) -> str:
        open_b, inner, close_b = m.group(1), m.group(2), m.group(3)
        close_map = {"[": "]", "(": ")", "{": "}"}
        expected_close = close_map.get(open_b, close_b)
        return f'{open_b}"{inner}"{expected_close}'

    # Match [text with spaces] / (text with spaces) / {text with spaces}
    # but not already-quoted ["..."] / ("...") / {"..."}
    pattern = re.compile(r'([\[\({])(?!")([^\]\)\}"]+\s[^\]\)\}"]*?)(?<!\")([\]\)}])')
    return pattern.sub(_quote_bracket, code)


def _fix_reserved_node_ids(code: str) -> str:
    """
    Prefix reserved words used as bare node IDs with an underscore.
    e.g.  end[End Process]  ->  _end[End Process]
    Only applies when the reserved word appears at the start of a line
    (after optional whitespace) or right after a connection arrow.
    """
    reserved_pattern = re.compile(
        r"(?<![A-Za-z0-9_])("
        + "|".join(re.escape(w) for w in _RESERVED_WORDS)
        + r")(\s*[\[\({>])",
        re.IGNORECASE,
    )

    def _replace(m: re.Match) -> str:
        return f"_{m.group(1)}{m.group(2)}"

    lines = code.splitlines()
    fixed = []
    if not lines:
        return code

    # Skip the first line (diagram type declaration)
    fixed.append(lines[0])
    for line in lines[1:]:
        if line.strip().startswith("%%"):
            fixed.append(line)
            continue
        fixed.append(reserved_pattern.sub(_replace, line))
    return "\n".join(fixed)


def _fix_node_ids_with_spaces(code: str) -> str:
    """
    Remove spaces/hyphens inside bare node IDs by camel-casing them.
    e.g.  My Node[...] -> MyNode[...]
          my-node[...] -> myNode[...]
    Only applies to flowchart/graph diagrams.
    """

    def _camel(m: re.Match) -> str:
        parts = re.split(r"[\s\-]+", m.group(1))
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        return f"{camel}{m.group(2)}"

    bad_id_pattern = re.compile(
        r"\b([A-Za-z0-9_]+(?:[\s\-][A-Za-z0-9_]+)+)(\s*[\[\({>])"
    )
    lines = code.splitlines()
    if not lines:
        return code

    fixed = [lines[0]]
    for line in lines[1:]:
        fixed.append(bad_id_pattern.sub(_camel, line))
    return "\n".join(fixed)


def _autofix_mermaid_block(code: str) -> str:
    """
    Apply all deterministic fixes to a single Mermaid block (without the
    surrounding ``` fences).  Order matters: fix IDs before labels.
    """
    lines = [ln.strip() for ln in code.strip().splitlines() if ln.strip()]
    if not lines:
        return code

    first_token = lines[0].split()[0].lower().rstrip(":")
    is_graph = first_token in {"graph", "flowchart"}

    fixed = code

    # Fix 1: reserved word node IDs
    fixed = _fix_reserved_node_ids(fixed)

    # Fix 2: node IDs with spaces/hyphens (graph/flowchart only)
    if is_graph:
        fixed = _fix_node_ids_with_spaces(fixed)

    # Fix 3: unquoted multi-word labels (graph/flowchart only)
    if is_graph:
        fixed = _fix_unquoted_labels(fixed)

    return fixed


def _apply_fixes_to_content(content: str) -> tuple[str, int]:
    """
    Find every Mermaid block in *content*, validate it, and — if it has
    errors — auto-fix it in-place.

    Returns ``(patched_content, num_fixed)`` where *num_fixed* is the number
    of blocks that were modified.
    """
    blocks = _extract_mermaid_blocks(content)
    if not blocks:
        return content, 0

    num_fixed = 0
    # Iterate in reverse so that index offsets stay valid after replacement
    for start, end, block_code in reversed(blocks):
        errors = _validate_mermaid_block(block_code)
        if not errors:
            continue

        fixed_code = _autofix_mermaid_block(block_code)
        new_fence = f"```mermaid\n{fixed_code}\n```"
        content = content[:start] + new_fence + content[end:]
        num_fixed += 1

    return content, num_fixed


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------
class MermaidValidationMiddleware(AgentMiddleware):
    """
    Validates Mermaid diagram syntax in every AI response and **directly
    patches** the AI message content with auto-corrected diagrams.

    No secondary model call is made — the LLM is invoked exactly once per
    user question regardless of diagram quality.

    Parameters
    ----------
    (none -kept signature compatible with the previous class)
    """

    # ------------------------------------------------------------------
    # Hook – runs after the model produces a response
    # ------------------------------------------------------------------

    def after_model(
        self, state: AgentState, runtime: Any
    ) -> dict[str, Any] | None:
        """
        Inspect the latest AI message for Mermaid blocks.

        If any block contains syntax errors, attempt to auto-fix them and
        replace the message content in-place.  Never jumps back to the model.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # The latest message should be the AI response
        latest = messages[-1]
        content: str = (
            latest.content
            if hasattr(latest, "content") and isinstance(latest.content, str)
            else ""
        )

        if not content:
            return None

        # Quick check: does it even contain a Mermaid block?
        if "```mermaid" not in content.lower():
            return None

        patched_content, num_fixed = _apply_fixes_to_content(content)

        if num_fixed == 0:
            logger.info("MermaidValidationMiddleware: all Mermaid blocks are valid.")
            return None

        logger.info(
            "MermaidValidationMiddleware: auto-fixed %d Mermaid block(s) in-place.",
            num_fixed,
        )

        # Build a patched copy of the latest message preserving all metadata
        patched_message = AIMessage(
            content=patched_content,
            additional_kwargs=getattr(latest, "additional_kwargs", {}),
            response_metadata=getattr(latest, "response_metadata", {}),
            id=getattr(latest, "id", None),
        )

        # Return updated messages list with the patched final message
        return {
            "messages": messages[:-1] + [patched_message],
        }