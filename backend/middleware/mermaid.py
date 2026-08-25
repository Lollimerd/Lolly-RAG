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
    "classdiagram",
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
    "c4context",
    "sankey-beta",
}

# Mermaid reserved words that must NOT be used as bare node IDs
_RESERVED_WORDS = {
    "graph",
    "flowchart",
    "subgraph",
    "end",
    "style",
    "classdef",
    "click",
    "call",
    "href",
    "linkstyle",
    "class",
    "direction",
}


# ---------------------------------------------------------------------------
# Extraction & Validation Helpers
# ---------------------------------------------------------------------------

def _extract_mermaid_blocks(text: str) -> list[tuple[int, int, str]]:
    """
    Find all ```mermaid ... ``` blocks in *text*.
    Also captures unclosed ```mermaid blocks at the end of *text*
    (e.g., from streaming cutoffs).

    Returns a list of (start_index, end_index, block_content) tuples.
    """
    results: list[tuple[int, int, str]] = []
    closed_spans: list[tuple[int, int]] = []

    # 1. Closed blocks
    for m in re.finditer(
        r"```mermaid\s*\n?(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    ):
        results.append((m.start(), m.end(), m.group(1)))
        closed_spans.append((m.start(), m.end()))

    # 2. Unclosed block at the end of text
    unclosed_match = re.search(
        r"```mermaid\s*\n?(.*?)$", text, flags=re.DOTALL | re.IGNORECASE
    )
    if unclosed_match:
        u_start = unclosed_match.start()
        is_already_closed = any(c_s <= u_start < c_e for c_s, c_e in closed_spans)
        if not is_already_closed:
            results.append((u_start, len(text), unclosed_match.group(1)))

    results.sort(key=lambda x: x[0])
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

    # 1. Check that the first non-comment line is a recognised diagram type
    first_non_comment = next((ln for ln in lines if not ln.startswith("%%")), "")
    if not first_non_comment:
        errors.append("Mermaid block contains only comments.")
        return errors

    first_token = first_non_comment.split()[0].lower().rstrip(":")
    if first_token not in _VALID_DIAGRAM_TYPES:
        errors.append(
            f"Unknown or missing diagram type on line: '{first_non_comment}'. "
            f"Expected one of: flowchart, sequenceDiagram, classDiagram, etc."
        )

    # 2. Check for reserved words used as bare node IDs
    reserved_node_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in _RESERVED_WORDS) + r")\s*[\[\({>]",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines[1:], start=2):
        if line.startswith("%%"):
            continue
        match = reserved_node_pattern.search(line)
        if match:
            errors.append(
                f"Line {i}: Reserved word '{match.group(1)}' used as a node ID."
            )

    # 3. Check for unclosed subgraphs
    if first_token in {"graph", "flowchart"}:
        open_subgraphs = 0
        for line in lines:
            if line.startswith("%%"):
                continue
            if re.match(r"^subgraph\b", line.strip(), re.IGNORECASE):
                open_subgraphs += 1
            elif re.match(r"^end\b", line.strip(), re.IGNORECASE):
                open_subgraphs = max(0, open_subgraphs - 1)
        if open_subgraphs > 0:
            errors.append(f"{open_subgraphs} unclosed subgraph block(s) detected.")

    return errors


# ---------------------------------------------------------------------------
# Auto-Fix Helpers
# ---------------------------------------------------------------------------

def _fix_missing_diagram_type(lines: list[str]) -> list[str]:
    """Ensure the diagram starts with a valid diagram header."""
    if not lines:
        return ["flowchart TD"]

    first_idx = 0
    while first_idx < len(lines) and lines[first_idx].strip().startswith("%%"):
        first_idx += 1

    if first_idx >= len(lines):
        return lines + ["flowchart TD"]

    first_line = lines[first_idx].strip()
    first_token = first_line.split()[0].lower().rstrip(":")

    if first_token not in _VALID_DIAGRAM_TYPES:
        # Prepend standard flowchart TD
        lines.insert(first_idx, "flowchart TD")

    return lines


def _fix_unclosed_brackets_and_quotes_line(line: str) -> str:
    """Balances quotes and brackets on a single line."""
    if line.strip().startswith("%%"):
        return line

    # Balance double quotes
    # Count unescaped quotes
    quote_count = len(re.findall(r'(?<!\\)"', line))
    if quote_count % 2 != 0:
        line += '"'

    # Balance brackets: [ ], ( ), { }
    # Count open vs close
    bracket_pairs = [("[", "]"), ("(", ")"), ("{", "}")]
    for open_b, close_b in bracket_pairs:
        open_cnt = line.count(open_b)
        close_cnt = line.count(close_b)
        if open_cnt > close_cnt:
            line += close_b * (open_cnt - close_cnt)

    return line


def _fix_dangling_connectors_line(line: str) -> str:
    """Removes trailing connector arrows with no target node."""
    stripped = line.strip()
    if stripped.startswith("%%") or not stripped:
        return line

    # Remove trailing connectors e.g., 'A -->', 'A ---', 'A ==>', 'A -.->', 'A -->|label|'
    dangling_patterns = [
        r"\s*(-->|---|==>|-\.->|-\.->|-->>|->>|->)\s*(\|[^|]*\|)?\s*$",
        r"\s*--\s*[^-\n]+\s*-->\s*$",
    ]
    for pat in dangling_patterns:
        stripped = re.sub(pat, "", stripped)

    # If line is only an arrow, clear it
    if stripped in {"-->", "---", "==>", "-.->", "-->>", "->>", "->", "|>"}:
        return ""

    return stripped


def _fix_unquoted_labels(code: str) -> str:
    """
    Wrap unquoted multi-word or special-char node labels in double-quotes.
    e.g.  NodeA[My Label]   ->  NodeA["My Label"]
          NodeA(My Label)   ->  NodeA("My Label")
          NodeA{Is Valid?}  ->  NodeA{"Is Valid?"}
    """
    def _quote_bracket(m: re.Match) -> str:
        open_b, inner, close_b = m.group(1), m.group(2), m.group(3)
        # Avoid double quoting if already quoted or contains database shape [(...)]
        if inner.startswith('"') and inner.endswith('"'):
            return f"{open_b}{inner}{close_b}"
        if inner.startswith("(") and inner.endswith(")"):
            # Shape like [(...)] -> [("...")]
            inner_sub = inner[1:-1].strip()
            if not (inner_sub.startswith('"') and inner_sub.endswith('"')):
                inner_sub = f'"{inner_sub}"'
            return f"{open_b}({inner_sub}){close_b}"
        # Escape internal unescaped quotes
        clean_inner = inner.replace('"', '\\"')
        return f'{open_b}"{clean_inner}"{close_b}'

    pattern = re.compile(
        r'([\[\({])(?!")([^\]\)\}"]*?[ \t:;,?!/\-*][^\]\)\}"]*?)(?<!\")([\]\)}])'
    )
    return pattern.sub(_quote_bracket, code)


def _fix_reserved_node_ids(code: str) -> str:
    """Prefix reserved words used as bare node IDs with an underscore."""
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

    fixed.append(lines[0])
    for line in lines[1:]:
        if line.strip().startswith("%%"):
            fixed.append(line)
            continue
        fixed.append(reserved_pattern.sub(_replace, line))
    return "\n".join(fixed)


def _fix_node_ids_with_spaces(code: str) -> str:
    """Remove spaces/hyphens inside bare node IDs by camel-casing them."""
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
        if line.strip().startswith("%%"):
            fixed.append(line)
            continue
        fixed.append(bad_id_pattern.sub(_camel, line))
    return "\n".join(fixed)


def _fix_unclosed_subgraphs(lines: list[str]) -> list[str]:
    """Ensure all subgraph and block declarations have matching 'end' statements."""
    first_non_comment = next((ln for ln in lines if not ln.startswith("%%")), "")
    first_token = first_non_comment.split()[0].lower().rstrip(":") if first_non_comment else ""

    open_blocks = 0
    block_starters = {
        "subgraph",
        "box",
        "loop",
        "alt",
        "opt",
        "par",
        "critical",
        "rect",
    }

    for line in lines:
        trimmed = line.strip()
        if trimmed.startswith("%%"):
            continue
        tokens = trimmed.split()
        if not tokens:
            continue
        leading_word = tokens[0].lower()
        if leading_word in block_starters:
            open_blocks += 1
        elif leading_word == "end":
            open_blocks = max(0, open_blocks - 1)

    # Append matching 'end' statements for any unclosed subgraphs/blocks
    for _ in range(open_blocks):
        lines.append("    end")

    return lines


def _autofix_mermaid_block(code: str) -> str:
    """
    Apply all deterministic fixes to a single Mermaid block (without code fences).
    Ensures the diagram is completely valid for Mermaid 10+.
    """
    raw_lines = [ln.rstrip() for ln in code.strip().splitlines() if ln.strip()]
    if not raw_lines:
        return "flowchart TD\n    A[\"Empty Diagram\"]"

    # Step 1: Ensure diagram type header
    lines = _fix_missing_diagram_type(raw_lines)

    # Step 2: Fix unclosed brackets, quotes, and dangling connectors line-by-line
    cleaned_lines: list[str] = []
    for line in lines:
        fixed_line = _fix_unclosed_brackets_and_quotes_line(line)
        fixed_line = _fix_dangling_connectors_line(fixed_line)
        if fixed_line.strip():
            cleaned_lines.append(fixed_line)

    # Step 3: Ensure all subgraphs/blocks have matching 'end'
    cleaned_lines = _fix_unclosed_subgraphs(cleaned_lines)

    fixed_code = "\n".join(cleaned_lines)

    # Step 4: Fix reserved words as node IDs
    fixed_code = _fix_reserved_node_ids(fixed_code)

    # Step 5: Fix node IDs with spaces
    first_token = cleaned_lines[0].split()[0].lower().rstrip(":") if cleaned_lines else ""
    if first_token in {"graph", "flowchart"}:
        fixed_code = _fix_node_ids_with_spaces(fixed_code)

    # Step 6: Quote multi-word labels
    fixed_code = _fix_unquoted_labels(fixed_code)

    return fixed_code


def _apply_fixes_to_content(content: str) -> tuple[str, int]:
    """
    Find every Mermaid block in *content*, validate it, and auto-fix it in-place.
    Also ensures all ```mermaid fences are properly closed.

    Returns ``(patched_content, num_fixed)`` where *num_fixed* is the number
    of blocks that were modified.
    """
    if not content or "```mermaid" not in content.lower():
        return content, 0

    blocks = _extract_mermaid_blocks(content)
    if not blocks:
        return content, 0

    num_fixed = 0
    # Iterate in reverse so that string offsets remain valid
    for start, end, block_code in reversed(blocks):
        fixed_code = _autofix_mermaid_block(block_code)
        new_fence = f"```mermaid\n{fixed_code}\n```"

        # Check if the block was actually changed or was unclosed
        original_segment = content[start:end]
        if original_segment != new_fence:
            content = content[:start] + new_fence + content[end:]
            num_fixed += 1

    return content, num_fixed


# ---------------------------------------------------------------------------
# Middleware Class
# ---------------------------------------------------------------------------

class MermaidValidationMiddleware(AgentMiddleware):
    """
    Validates Mermaid diagram syntax in every AI response and directly
    patches the AI message content with auto-corrected diagrams.
    """

    def after_model(
        self, state: AgentState, runtime: Any
    ) -> dict[str, Any] | None:
        """
        Inspect the latest AI message for Mermaid blocks and auto-fix in-place.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        latest = messages[-1]
        content: str = (
            latest.content
            if hasattr(latest, "content") and isinstance(latest.content, str)
            else ""
        )

        if not content or "```mermaid" not in content.lower():
            return None

        patched_content, num_fixed = _apply_fixes_to_content(content)

        if num_fixed == 0:
            logger.info("MermaidValidationMiddleware: all Mermaid blocks are valid.")
            return None

        logger.info(
            "MermaidValidationMiddleware: auto-fixed %d Mermaid block(s) in-place.",
            num_fixed,
        )

        patched_message = AIMessage(
            content=patched_content,
            additional_kwargs=getattr(latest, "additional_kwargs", {}),
            response_metadata=getattr(latest, "response_metadata", {}),
            id=getattr(latest, "id", None),
        )

        return {
            "messages": messages[:-1] + [patched_message],
        }