"""
backend/middleware/mermaid.py
-----------------------------
Mermaid Validation and Auto-Correction Middleware for LangChain Agents.

Features:
1. Syntax validation before rendering: Comprehensive regex and AST-like structural
   error detection (headers, delimiters, unclosed subgraphs, arrow connectors, reserved words).
2. Auto-correction using LLM: Deterministic prompt with zero temperature to heal complex
   diagram errors when fast rule-based repairs are insufficient.
3. Configurable retry logic: Multi-pass progressive healing loop with configurable attempt limits.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex Patterns & Constants
# ---------------------------------------------------------------------------

_FENCE_REGEX = re.compile(
    r"```(?:mermaid|MERMAID|Mermaid)[ \t]*\r?\n?([\s\S]*?)```",
    re.IGNORECASE,
)

_UNCLOSED_FENCE_REGEX = re.compile(
    r"```(?:mermaid|MERMAID|Mermaid)[ \t]*\r?\n?([\s\S]+)$",
    re.IGNORECASE,
)

VALID_HEADER_PATTERNS = (
    r"^flowchart\s+(?:td|tb|bt|rl|lr)\b",
    r"^graph\s+(?:td|tb|bt|rl|lr)\b",
    r"^sequencediagram\b",
    r"^classdiagram(?:-v2)?\b",
    r"^statediagram(?:-v2)?\b",
    r"^erdiagram\b",
    r"^journey\b",
    r"^gantt\b",
    r"^pie\b",
    r"^gitgraph\b",
    r"^mindmap\b",
    r"^quadrantchart\b",
    r"^timeline\b",
    r"^sankey-beta\b",
    r"^requirementdiagram\b",
    r"^packet-beta\b",
    r"^block-beta\b",
    r"^xychart-beta\b",
    r"^architecture-beta\b",
    r"^c4(?:context|container|component|dynamic|deployment)\b",
    r"^zenuml\b",
)

HEADER_CORRECTIONS = {
    "flowcharttd": "flowchart TD",
    "flowchartlr": "flowchart LR",
    "flowcharttb": "flowchart TB",
    "flowchartbt": "flowchart BT",
    "flowchartrl": "flowchart RL",
    "graphtd": "graph TD",
    "graphlr": "graph LR",
    "graphtb": "graph TB",
    "graphbt": "graph BT",
    "graphrl": "graph RL",
    "flow-chart td": "flowchart TD",
    "flow-chart lr": "flowchart LR",
    "flow-chart tb": "flowchart TB",
    "flow-chart": "flowchart TD",
    "flowchart": "flowchart TD",
    "graph": "graph TD",
    "sequence-diagram": "sequenceDiagram",
    "sequence diagram": "sequenceDiagram",
    "sequencediagram": "sequenceDiagram",
    "class-diagram": "classDiagram",
    "class diagram": "classDiagram",
    "classdiagram": "classDiagram",
    "state-diagram": "stateDiagram-v2",
    "state diagram": "stateDiagram-v2",
    "statediagram": "stateDiagram-v2",
    "er-diagram": "erDiagram",
    "er diagram": "erDiagram",
    "erdiagram": "erDiagram",
    "git-graph": "gitGraph",
    "git graph": "gitGraph",
    "gitgraph": "gitGraph",
}

RESERVED_NODE_WORDS = {
    "end",
    "graph",
    "flowchart",
    "subgraph",
    "click",
    "style",
    "class",
    "classDef",
    "linkStyle",
    "default",
    "interpolate",
    "call",
}

SHAPE_SPECIFICATIONS = [
    ("[((", "))]", '[(( "', '" ))]'),
    ("[([", "])]", '[([ "', '" ])]'),
    ("[/", "/]", '[/ "', '" /]'),
    ("[\\", "\\]", '[\\ "', '" \\]'),
    ("[/", "\\]", '[/ "', '" \\]'),
    ("[\\", "/]", '[\\ "', '" /]'),
    ("[(((", "))) ]", '[((( "', '" )))]'),
    ("[(((", ")))", '[((( "', '" )))]'),
    ("[((", "))", '[(( "', '" ))]'),
    ("[[", "]]", '[["', '"]]'),
    ("[([", "])", '([["', '"]])'),
    ("[(((", ")))", '([(("', '"))])'),
    ("[(", ")]", '[("', '")]'),
    ("([", "])", '(["', '"])'),
    ("{{", "}}", '{{"', '"}}'),
    ("(((", ")))", '((("', '")))'),
    ("((", "))", '(("', '"))'),
    (">", "]", '>"', '"]'),
    ("{", "}", '{"', '"}'),
    ("(", ")", '("', '")'),
    ("[", "]", '["', '"]'),
]


# ---------------------------------------------------------------------------
# Quote-Aware Parsing Helpers
# ---------------------------------------------------------------------------

def _split_by_quotes(line: str) -> list[tuple[str, bool]]:
    """Split line into alternating (unquoted, quoted) segments."""
    segments: list[tuple[str, bool]] = []
    in_quote = False
    current: list[str] = []
    escaped = False

    for ch in line:
        if ch == "\\" and not escaped:
            escaped = True
            current.append(ch)
            continue

        if ch == '"' and not escaped:
            if in_quote:
                current.append(ch)
                segments.append(("".join(current), True))
                current = []
                in_quote = False
            else:
                if current:
                    segments.append(("".join(current), False))
                    current = []
                current.append(ch)
                in_quote = True
        else:
            current.append(ch)
        escaped = False

    if current:
        segments.append(("".join(current), in_quote))

    return segments


def _find_shape_close(text: str, start_idx: int, open_delim: str, close_delim: str) -> int:
    """Find matching shape close delimiter respecting nesting."""
    depth = 0
    balance_open = None
    balance_close = None
    if open_delim in ("(", "((", "(((") and close_delim in (")", "))", ")))"):
        balance_open, balance_close = "(", ")"
    elif open_delim in ("[", "[[") and close_delim in ("]", "]]"):
        balance_open, balance_close = "[", "]"
    elif open_delim in ("{", "{{") and close_delim in ("}", "}}"):
        balance_open, balance_close = "{", "}"

    i = start_idx
    delim_len = len(close_delim)
    while i <= len(text) - delim_len:
        if text[i : i + delim_len] == close_delim and depth == 0:
            return i
        if balance_open and text[i] == balance_open:
            depth += 1
        elif balance_close and text[i] == balance_close:
            if depth > 0:
                depth -= 1
        i += 1

    return text.find(close_delim, start_idx)


# ---------------------------------------------------------------------------
# 1. Syntax Validation Before Rendering
# ---------------------------------------------------------------------------

def find_mermaid_errors(code: str) -> list[str]:
    """
    Scans a Mermaid diagram string for syntax, structural, and delimiter errors.
    Returns a list of human-readable error descriptions.
    """
    errors: list[str] = []
    raw_lines = code.strip().splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    if not lines:
        return ["Empty or blank Mermaid diagram"]

    # Filter out YAML frontmatter or comments to find first effective line
    first_code_line = ""
    in_frontmatter = False
    for line in lines:
        if line == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if line.startswith("%%"):
            continue
        first_code_line = line
        break

    if not first_code_line:
        return ["No valid diagram content found (comments or frontmatter only)"]

    # 1. Header Validation
    header_lower = first_code_line.lower().strip()
    has_valid_header = any(re.match(pattern, header_lower, re.IGNORECASE) for pattern in VALID_HEADER_PATTERNS)

    if not has_valid_header:
        errors.append(
            f"Missing or malformed diagram header declaration on line 1: '{first_code_line}'"
        )

    # 2. Delimiter and Subgraph Balance
    sq_open = sq_close = 0
    paren_open = paren_close = 0
    curly_open = curly_close = 0
    pipe_count = 0
    subgraph_open = 0
    subgraph_closed = 0
    seq_block_open = 0
    seq_block_closed = 0

    is_seq_diagram = "sequencediagram" in header_lower

    for idx, line in enumerate(lines, 1):
        if line.startswith("%%") or line == "---":
            continue

        line_clean = line.strip()

        # Non-diagram commentary detection
        if (
            line_clean.startswith(("- ", "* ", "1. ", "2. ", "3. ", "Note: ", "Here is ", "Explanation:"))
            and not is_seq_diagram
            and "-->" not in line_clean
            and "[" not in line_clean
        ):
            errors.append(f"Line {idx}: Non-diagram commentary or markdown list detected: '{line_clean}'")

        # Track subgraph balance
        if re.match(r"^subgraph\b", line_clean, re.IGNORECASE):
            subgraph_open += 1
        elif line_clean.lower() == "end" or re.match(r"^end\b", line_clean, re.IGNORECASE):
            subgraph_closed += 1
            if is_seq_diagram:
                seq_block_closed += 1

        # Track sequence diagram block keywords
        if is_seq_diagram and re.match(r"^(loop|alt|opt|par|critical|rect)\b", line_clean, re.IGNORECASE):
            seq_block_open += 1

        if not is_seq_diagram:
            for seg, is_q in _split_by_quotes(line):
                if not is_q:
                    # Single dash arrow detection
                    if re.search(r"(?<![\-\.\=\>\<])\s*->\s*(?![\-\.\=\>\<])", seg):
                        errors.append(f"Line {idx}: Single dash arrow '->' detected; flowchart requires '-->' or '--o'")
                    # Reserved keyword node IDs
                    if re.search(r"\b(end|graph|flowchart|subgraph)\s*(?:\[|\(|\{|\>|\-\-|\=\=)", seg, re.IGNORECASE):
                        if not re.match(r"^(subgraph|graph|flowchart)\b", line_clean, re.IGNORECASE):
                            errors.append(f"Line {idx}: Reserved keyword used as standalone node identifier")
                    # Unquoted labels with special characters
                    unquoted_paren_match = re.search(r"\w+\s*\[([^\"\]]*[\(\):,][^\"\]]*)\]", seg)
                    if unquoted_paren_match:
                        errors.append(f"Line {idx}: Unquoted node label with special characters: '[{unquoted_paren_match.group(1)}]'")

        flag_count = len(re.findall(r"\b\w+\s*>\s*(?:\"[^\"]*\"|[^\]]+)\]", line_clean))
        sq_open += flag_count

        for seg, is_q in _split_by_quotes(line):
            if is_q:
                continue
            sq_open += seg.count("[")
            sq_close += seg.count("]")
            paren_open += seg.count("(")
            paren_close += seg.count(")")
            curly_open += seg.count("{")
            curly_close += seg.count("}")
            pipe_count += seg.count("|")

    if sq_open != sq_close:
        errors.append(f"Unbalanced square brackets: {sq_open} '[' vs {sq_close} ']'")
    if paren_open != paren_close:
        errors.append(f"Unbalanced parentheses: {paren_open} '(' vs {paren_close} ')'")
    if curly_open != curly_close:
        errors.append(f"Unbalanced curly braces: {curly_open} '{{' vs {curly_close} '}}'")
    if pipe_count % 2 != 0:
        errors.append(f"Unbalanced link label pipe symbols '|' ({pipe_count} total)")

    if subgraph_open > subgraph_closed:
        errors.append(f"Unclosed subgraph block: {subgraph_open} 'subgraph' vs {subgraph_closed} 'end'")
    elif subgraph_closed > subgraph_open and not is_seq_diagram:
        errors.append(f"Orphaned 'end' statement without matching 'subgraph'")

    if is_seq_diagram and seq_block_open > seq_block_closed:
        errors.append(f"Unclosed sequence diagram block: {seq_block_open} open blocks vs {seq_block_closed} 'end'")

    return errors


def validate_mermaid(code: str) -> tuple[bool, list[str]]:
    """
    Validates a Mermaid diagram string.
    Returns (True, []) if valid, or (False, [error_messages]) if errors were detected.
    """
    errors = find_mermaid_errors(code)
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Rule-Based Progressive Healing Passes
# ---------------------------------------------------------------------------

def _fix_unquoted_labels_on_line(line: str) -> str:
    """Wraps unquoted labels for all Mermaid shapes in double quotes."""
    line_clean = line.strip()
    if not line_clean or line_clean.startswith("%%") or line_clean.startswith("---"):
        return line

    first_token = line_clean.split()[0].lower()
    if first_token in ("flowchart", "graph", "sequencediagram", "classdiagram", "statediagram", "direction"):
        return line

    specs = sorted(SHAPE_SPECIFICATIONS, key=lambda s: len(s[0]), reverse=True)
    openings_pattern = "|".join(re.escape(s[0]) for s in specs)
    node_pattern = re.compile(rf"(\b[a-zA-Z0-9_]+)\s*({openings_pattern})")

    pos = 0
    result: list[str] = []

    while pos < len(line):
        match = node_pattern.search(line, pos)
        if not match:
            result.append(line[pos:])
            break

        result.append(line[pos : match.start()])
        node_id = match.group(1)
        matched_open = match.group(2)

        spec = next((s for s in specs if s[0] == matched_open), None)
        if not spec:
            result.append(line[match.start() : match.end()])
            pos = match.end()
            continue

        open_delim, close_delim, wrap_prefix, wrap_suffix = spec
        start_content = match.end()

        close_idx = _find_shape_close(line, start_content, open_delim, close_delim)
        if close_idx != -1:
            raw_content = line[start_content:close_idx].strip()
            if raw_content.startswith('"') and raw_content.endswith('"') and len(raw_content) >= 2:
                inner = raw_content[1:-1]
                if '"' in inner:
                    clean_inner = inner.replace('"', "'")
                    result.append(f"{node_id}{open_delim}\"{clean_inner}\"{close_delim}")
                else:
                    result.append(f"{node_id}{open_delim}{raw_content}{close_delim}")
            else:
                clean_content = raw_content.replace('"', "'")
                result.append(f"{node_id}{wrap_prefix}{clean_content}{wrap_suffix}")
            pos = close_idx + len(close_delim)
        else:
            result.append(line[match.start() : match.end()])
            pos = match.end()

    return "".join(result)


def _repair_pass_1(code: str) -> str:
    """Pass 1: Header, delimiter, and label quote normalization."""
    code = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", code)
    code = re.sub(r"^```(?:mermaid|MERMAID|Mermaid)?\s*$", "", code, flags=re.MULTILINE | re.IGNORECASE)
    code = re.sub(r"^```\s*$", "", code, flags=re.MULTILINE)

    lines = [line.rstrip() for line in code.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return "flowchart TD\n    Start([Start]) --> End([End])"

    header_idx = -1
    in_frontmatter = False
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if line_clean == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter or line_clean.startswith("%%") or not line_clean:
            continue
        header_idx = i
        break

    if header_idx != -1:
        raw_header = lines[header_idx].strip()
        raw_header_lower = raw_header.lower()

        is_valid_header = any(re.match(p, raw_header_lower, re.IGNORECASE) for p in VALID_HEADER_PATTERNS)
        if not is_valid_header:
            if raw_header_lower in HEADER_CORRECTIONS:
                lines[header_idx] = HEADER_CORRECTIONS[raw_header_lower]
            else:
                first_tok = raw_header_lower.split()[0] if raw_header_lower.split() else ""
                if first_tok in HEADER_CORRECTIONS:
                    rest = raw_header[len(first_tok):].strip()
                    lines[header_idx] = f"{HEADER_CORRECTIONS[first_tok]} {rest}".strip()
                elif not any(re.match(p, raw_header_lower, re.IGNORECASE) for p in VALID_HEADER_PATTERNS):
                    lines.insert(header_idx, "flowchart TD")
    else:
        lines.insert(0, "flowchart TD")

    repaired_lines = [_fix_unquoted_labels_on_line(line) for line in lines]
    return "\n".join(repaired_lines)


def _repair_pass_2(code: str) -> str:
    """Pass 2: Identifier, connector, and reserved keyword sanitization."""
    lines = code.splitlines()
    repaired: list[str] = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("%%") or line_clean.startswith("---"):
            repaired.append(line)
            continue

        # Subgraph declaration normalization
        subgraph_match = re.match(r"^subgraph\s+([^\s\[\]]+(?:\s+[^\s\[\]]+)+)\s*$", line_clean, re.IGNORECASE)
        if subgraph_match and not ("[" in line_clean and "]" in line_clean):
            title = subgraph_match.group(1).strip()
            node_id = re.sub(r"[^\w]", "_", title).strip("_")
            repaired.append(f'    subgraph {node_id} ["{title}"]')
            continue

        segments = _split_by_quotes(line)
        new_segments: list[str] = []

        for seg_text, is_quoted in segments:
            if is_quoted:
                new_segments.append(seg_text)
                continue

            seg = seg_text
            # 1. Single dash arrows
            seg = re.sub(r"(?<![\-\.\=\>\<])\s*->\s*(?![\-\.\=\>\<])", " --> ", seg)
            # 1.5 Normalize spacing around valid arrows
            seg = re.sub(r"(?<!\s)(-->|---|==>|-\.->)", r" \1", seg)
            seg = re.sub(r"(-->|---|==>|-\.->)(?![\|\s])", r"\1 ", seg)
            # 2. 'A -- label --> B' -> 'A -->|label| B'
            seg = re.sub(r"\s+--\s*([^\|\-\>\n][^\-\>\n]*?)\s*-->\s*", r" -->|\1| ", seg)
            seg = re.sub(r"\s+--\s*([^\|\-\n][^\-\n]*?)\s*--\s*", r" ---|\1| ", seg)
            seg = re.sub(r"\s+--\s*([a-zA-Z0-9_\s]{2,})\s+(\w+)\b", r" -->|\1| \2", seg)
            # 3. Unclosed link pipes
            seg = re.sub(r"(-->|---|==>|-\.->)\|([^\|\n]+?)\s+([A-Za-z0-9_]+)\b", r"\1|\2| \3", seg)

            # 4. Reserved keyword node IDs
            if line_clean.lower() != "end":
                for res_word in RESERVED_NODE_WORDS:
                    seg = re.sub(rf"\b({res_word})\s*([\[\(\{{\>])", rf"node_\1\2", seg, flags=re.IGNORECASE)
                    seg = re.sub(rf"\b({res_word})\s*(-->|---|==>|-\.->)", rf"node_\1 \2", seg, flags=re.IGNORECASE)
                    seg = re.sub(rf"(-->|---|==>|-\.->)\s*\b({res_word})\b(?!\s*[\w\[\(\{{\>])", rf"\1 node_\2", seg, flags=re.IGNORECASE)

            # 5. Hyphenated node IDs outside quotes
            seg = re.sub(r"\b([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)\s*([\[\(\{{\>])", r"\1_\2\3", seg)
            seg = re.sub(r"\b([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)\s+(-->|---|==>|-\.->)", r"\1_\2 \3", seg)
            seg = re.sub(r"(-->|---|==>|-\.->)\s+([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)\b", r"\1 \2_\3", seg)

            new_segments.append(seg)

        repaired.append("".join(new_segments))

    return "\n".join(repaired)


def _repair_pass_3(code: str) -> str:
    """Pass 3: Subgraph balancing and sequence block closure."""
    lines = code.splitlines()
    repaired_lines: list[str] = []

    subgraph_depth = 0
    seq_depth = 0
    is_seq = False

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            repaired_lines.append("")
            continue

        if "sequencediagram" in line_clean.lower():
            is_seq = True

        if line_clean.startswith("%%") or line_clean.startswith("---"):
            repaired_lines.append(line)
            continue

        if (
            line_clean.startswith(("- ", "* ", "1. ", "2. ", "3. ", "Note: ", "Here is ", "Explanation:"))
            and not is_seq
            and "-->" not in line_clean
            and "[" not in line_clean
        ):
            repaired_lines.append(f"    %% {line_clean}")
            continue

        if re.match(r"^subgraph\b", line_clean, re.IGNORECASE):
            subgraph_depth += 1
            repaired_lines.append(line)
            continue
        elif line_clean.lower() == "end" or re.match(r"^end\b", line_clean, re.IGNORECASE):
            if is_seq and seq_depth > 0:
                seq_depth -= 1
                repaired_lines.append("    end")
            elif subgraph_depth > 0:
                subgraph_depth -= 1
                repaired_lines.append("    end")
            continue

        if is_seq and re.match(r"^(loop|alt|opt|par|critical|rect)\b", line_clean, re.IGNORECASE):
            seq_depth += 1
            repaired_lines.append(line)
            continue

        repaired_lines.append(line)

    while subgraph_depth > 0:
        repaired_lines.append("    end")
        subgraph_depth -= 1

    while seq_depth > 0:
        repaired_lines.append("    end")
        seq_depth -= 1

    final_code = "\n".join(repaired_lines)
    if not any(re.match(p, final_code.lower().strip(), re.IGNORECASE) for p in VALID_HEADER_PATTERNS):
        final_code = "flowchart TD\n" + final_code

    return final_code


# ---------------------------------------------------------------------------
# 2. Auto-Correction Using LLM
# ---------------------------------------------------------------------------

_LLM_CORRECTION_PROMPT = """You are a specialized Mermaid diagram syntax fixer.
Fix the following malformed Mermaid diagram so that it is 100% syntactically valid and renders cleanly.

Detected Syntax Errors:
{errors}

Original Diagram:
```mermaid
{code}
```

STRICT RULES:
1. Fix all syntax errors, unbalanced brackets, broken arrows, and invalid headers.
2. Every node label text must be enclosed in double quotes: e.g. Node1["Title (with details)"] or DB[("Database")].
3. Node IDs must be alphanumeric and underscores only ([a-zA-Z0-9_]+). No reserved keywords (end, subgraph, graph, click) as standalone IDs.
4. Every subgraph must have a matching 'end'.
5. Ensure spaces exist around all arrows (e.g. 'A --> B' instead of 'A-->B').
6. Output ONLY the raw Mermaid diagram syntax. Do NOT wrap in ```mermaid markdown fences. Do NOT add any explanations or notes.
"""


def _get_fixer_llm(llm: Optional[Any] = None) -> Any:
    """Returns a deterministic LLM instance for Mermaid auto-correction."""
    if llm is not None:
        return llm
    try:
        from setup.init_config import cypher_LLM
        return cypher_LLM()
    except Exception:
        try:
            from langchain_ollama import ChatOllama
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://local-ai:11434")
            return ChatOllama(
                model="qwen3.5:4b",
                base_url=ollama_url,
                temperature=0.0,
                reasoning=False,
                num_predict=1024,
                tags=["mermaid_fixer"],
            )
        except Exception as e:
            logger.warning(f"Could not initialize default ChatOllama for Mermaid fixer: {e}")
            return None


def _llm_fix_mermaid(code: str, errors: list[str], llm: Optional[Any] = None) -> str:
    """Invokes LLM to auto-correct malformed Mermaid diagram."""
    fixer_llm = _get_fixer_llm(llm)
    if fixer_llm is None:
        return code

    error_bullet_list = "\n".join(f"- {err}" for err in errors)
    prompt_text = _LLM_CORRECTION_PROMPT.format(errors=error_bullet_list, code=code.strip())

    try:
        response = fixer_llm.invoke([
            SystemMessage(content="You are an expert Mermaid diagram syntax repair engine. Output ONLY valid Mermaid syntax."),
            HumanMessage(content=prompt_text),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        # Extract content from fences if LLM wrapped it
        if "```" in content:
            match = _FENCE_REGEX.search(content)
            if match:
                content = match.group(1)
            else:
                content = re.sub(r"^```(?:mermaid|MERMAID)?\s*$", "", content, flags=re.MULTILINE)
                content = re.sub(r"^```\s*$", "", content, flags=re.MULTILINE)

        cleaned = content.strip()
        logger.info("LLM auto-correction generated candidate Mermaid syntax.")
        return cleaned if cleaned else code
    except Exception as exc:
        logger.warning(f"LLM Mermaid auto-correction call failed: {exc}")
        return code


# ---------------------------------------------------------------------------
# 3. Retry Logic with Configurable Attempts
# ---------------------------------------------------------------------------

def repair_mermaid(
    code: str,
    max_retries: int = 3,
    use_llm: bool = True,
    llm: Optional[Any] = None,
) -> tuple[str, bool, list[str]]:
    """
    Progressively validates and repairs a Mermaid diagram block across up to `max_retries` attempts.
    Combines multi-pass rule repairs with LLM auto-correction on persistent syntax failures.

    Parameters:
        code (str): Raw Mermaid diagram string.
        max_retries (int): Maximum repair attempt passes (default: 3).
        use_llm (bool): Whether to invoke LLM auto-correction if rule-based passes fail.
        llm (Optional[Any]): Optional LLM instance for auto-correction.

    Returns:
        tuple[str, bool, list[str]]: (repaired_code, was_modified, list_of_fixes_applied)
    """
    clean_code = code.strip()
    if not clean_code:
        return "flowchart TD\n    Start([Start]) --> End([End])", True, ["Initialized empty diagram"]

    log: list[str] = []
    current_code = clean_code
    was_modified = False

    initial_errors = find_mermaid_errors(current_code)
    if not initial_errors:
        return current_code, False, []

    log.append(f"Initial errors ({len(initial_errors)}): {initial_errors[0]}")

    for attempt in range(1, max_retries + 1):
        prev_code = current_code

        # Attempt 1: Headers & Label Quotes
        if attempt == 1:
            current_code = _repair_pass_1(current_code)
            log.append(f"Attempt 1/{max_retries}: Normalized headers and shape label quotes.")

        # Attempt 2: Node IDs & Connectors
        elif attempt == 2:
            current_code = _repair_pass_2(current_code)
            log.append(f"Attempt 2/{max_retries}: Sanitized node identifiers, arrows, and reserved words.")

        # Attempt 3+: Subgraphs & LLM Auto-Correction Pass
        elif attempt >= 3:
            current_code = _repair_pass_3(current_code)
            log.append(f"Attempt {attempt}/{max_retries}: Balanced subgraphs and sequence blocks.")

            # If still invalid, trigger LLM auto-correction
            remaining_before_llm = find_mermaid_errors(current_code)
            if remaining_before_llm and use_llm:
                log.append(f"Attempt {attempt}/{max_retries}: Triggering LLM auto-correction pass...")
                llm_fixed = _llm_fix_mermaid(current_code, remaining_before_llm, llm=llm)
                # Run pass 1 & 2 on LLM output to ensure clean formatting
                llm_fixed = _repair_pass_1(llm_fixed)
                llm_fixed = _repair_pass_2(llm_fixed)
                if not find_mermaid_errors(llm_fixed) or len(find_mermaid_errors(llm_fixed)) < len(remaining_before_llm):
                    current_code = llm_fixed
                    log.append(f"Attempt {attempt}/{max_retries}: LLM auto-correction improved syntax.")

        if current_code != prev_code:
            was_modified = True

        remaining_errors = find_mermaid_errors(current_code)
        if not remaining_errors:
            log.append(f"Validation succeeded after Attempt {attempt}/{max_retries}.")
            return current_code, was_modified, log

        log.append(f"Attempt {attempt}/{max_retries} finished with remaining issue: {remaining_errors[0]}")

    return current_code, was_modified, log


# ---------------------------------------------------------------------------
# Content Processing & Mermaid Block Replacement
# ---------------------------------------------------------------------------

def extract_mermaid_blocks(content: str) -> list[tuple[str, int, int]]:
    """Extracts all Mermaid diagram blocks from a text string."""
    blocks: list[tuple[str, int, int]] = []
    for match in _FENCE_REGEX.finditer(content):
        blocks.append((match.group(1), match.start(), match.end()))
    return blocks


def _apply_fixes_to_content(
    content: str,
    max_retries: int = 3,
    use_llm: bool = True,
    llm: Optional[Any] = None,
) -> tuple[str, bool]:
    """Scans content for ```mermaid ... ``` blocks, validates syntax, and auto-corrects them."""
    if not content or "```" not in content:
        return content, False

    any_modified = False

    def _replace_block(match: re.Match) -> str:
        nonlocal any_modified
        raw_diagram = match.group(1)
        repaired, modified, _ = repair_mermaid(raw_diagram, max_retries=max_retries, use_llm=use_llm, llm=llm)
        if modified:
            any_modified = True
        return f"```mermaid\n{repaired}\n```"

    result = _FENCE_REGEX.sub(_replace_block, content)

    unclosed_match = _UNCLOSED_FENCE_REGEX.search(result)
    if unclosed_match and not result.rstrip().endswith("```"):
        raw_diagram = unclosed_match.group(1)
        repaired, _, _ = repair_mermaid(raw_diagram, max_retries=max_retries, use_llm=use_llm, llm=llm)
        result = _UNCLOSED_FENCE_REGEX.sub(f"```mermaid\n{repaired}\n```", result)
        any_modified = True

    return result, any_modified


# ---------------------------------------------------------------------------
# LangChain Agent Middleware Class
# ---------------------------------------------------------------------------

class MermaidValidationMiddleware(AgentMiddleware[Any, Any, Any]):
    """
    AgentMiddleware that intercepts model responses, validates Mermaid syntax before
    rendering, applies LLM auto-correction when needed, and executes configurable retry attempts.
    """

    name: str = "MermaidValidation"

    def __init__(
        self,
        max_retries: int = 3,
        use_llm: bool = True,
        llm: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.max_retries = max_retries
        self.use_llm = use_llm
        self.llm = llm

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        response = handler(request)
        return self._process_model_response(response)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> ModelResponse[Any] | AIMessage:
        response = await handler(request)
        return self._process_model_response(response)

    def _process_model_response(self, response: Any) -> Any:
        if isinstance(response, ModelResponse):
            for msg in response.result:
                self._sanitize_message_content(msg)
            return response
        elif isinstance(response, AIMessage):
            self._sanitize_message_content(response)
            return response
        return response

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._process_state(state)

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._process_state(state)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._process_state(state)

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._process_state(state)

    def _sanitize_message_content(self, msg: BaseMessage) -> bool:
        if not hasattr(msg, "content"):
            return False

        content = msg.content
        if isinstance(content, str) and "```" in content:
            fixed_content, modified = _apply_fixes_to_content(
                content, max_retries=self.max_retries, use_llm=self.use_llm, llm=self.llm
            )
            if modified:
                msg.content = fixed_content
                return True
        elif isinstance(content, list):
            modified_any = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part_text = part.get("text", "")
                    if "```" in part_text:
                        fixed_text, mod = _apply_fixes_to_content(
                            part_text, max_retries=self.max_retries, use_llm=self.use_llm, llm=self.llm
                        )
                        if mod:
                            part["text"] = fixed_text
                            modified_any = True
            return modified_any

        return False

    def _process_state(self, state: Any) -> dict[str, Any] | None:
        if not isinstance(state, dict):
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        state_modified = False
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                if self._sanitize_message_content(msg):
                    state_modified = True
                break

        if state_modified:
            logger.info("MermaidValidationMiddleware auto-corrected diagrams in agent state")
            return {"messages": messages}
        return None
