import html
import logging
import re
import uuid
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

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


def autofix_mermaid_code(code: str) -> str:
    """
    Applies deterministic syntax repairs to Mermaid diagram source code.
    Fixes missing headers, unclosed subgraphs, dangling arrows, unclosed brackets/quotes,
    and reserved-word node IDs.
    """
    raw_lines = [ln.rstrip() for ln in code.strip().splitlines() if ln.strip()]
    if not raw_lines:
        return "flowchart TD\n    A[\"Empty Diagram\"]"

    # 1. Ensure valid diagram header
    first_non_comment_idx = 0
    while first_non_comment_idx < len(raw_lines) and raw_lines[first_non_comment_idx].strip().startswith("%%"):
        first_non_comment_idx += 1

    if first_non_comment_idx >= len(raw_lines):
        raw_lines.append("flowchart TD")
    else:
        first_token = raw_lines[first_non_comment_idx].strip().split()[0].lower().rstrip(":")
        if first_token not in _VALID_DIAGRAM_TYPES:
            raw_lines.insert(first_non_comment_idx, "flowchart TD")

    # 2. Fix unclosed quotes/brackets & dangling connectors per line
    cleaned_lines = []
    for line in raw_lines:
        trimmed = line.strip()
        if trimmed.startswith("%%"):
            cleaned_lines.append(line)
            continue

        # Balance double quotes on line
        quote_count = len(re.findall(r'(?<!\\)"', trimmed))
        if quote_count % 2 != 0:
            trimmed += '"'

        # Balance brackets: [ ], ( ), { }
        for open_b, close_b in [("[", "]"), ("(", ")"), ("{", "}")]:
            open_c = trimmed.count(open_b)
            close_c = trimmed.count(close_b)
            if open_c > close_c:
                trimmed += close_b * (open_c - close_c)

        # Remove dangling connectors at line end (e.g. 'A -->', 'A ---', 'A ==>')
        dangling_patterns = [
            r"\s*(-->|---|==>|-\.->|-->>|->>|->)\s*(\|[^|]*\|)?\s*$",
            r"\s*--\s*[^-\n]+\s*-->\s*$",
        ]
        for pat in dangling_patterns:
            trimmed = re.sub(pat, "", trimmed)

        if trimmed and trimmed not in {"-->", "---", "==>", "-.->", "-->>", "->>", "->"}:
            cleaned_lines.append(trimmed)

    # 3. Ensure matching 'end' statements for open subgraphs/blocks
    open_blocks = 0
    block_starters = {"subgraph", "box", "loop", "alt", "opt", "par", "critical", "rect"}
    for line in cleaned_lines:
        if line.strip().startswith("%%"):
            continue
        tokens = line.strip().split()
        if tokens:
            leading = tokens[0].lower()
            if leading in block_starters:
                open_blocks += 1
            elif leading == "end":
                open_blocks = max(0, open_blocks - 1)

    for _ in range(open_blocks):
        cleaned_lines.append("    end")

    fixed = "\n".join(cleaned_lines)

    # 4. Fix reserved words used as bare node IDs
    reserved_pat = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(w) for w in _RESERVED_WORDS) + r")(\s*[\[\({>])",
        re.IGNORECASE,
    )
    fixed = reserved_pat.sub(r"_\1\2", fixed)

    # 5. CamelCase node IDs containing spaces/hyphens
    first_token = cleaned_lines[0].split()[0].lower().rstrip(":") if cleaned_lines else ""
    if first_token in {"graph", "flowchart"}:
        def _camel(m: re.Match) -> str:
            parts = re.split(r"[\s\-]+", m.group(1))
            camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
            return f"{camel}{m.group(2)}"

        bad_id_pattern = re.compile(r"\b([A-Za-z0-9_]+(?:[\s\-][A-Za-z0-9_]+)+)(\s*[\[\({>])")
        fixed = bad_id_pattern.sub(_camel, fixed)

    # 6. Wrap unquoted multi-word node labels in double-quotes
    def _quote_bracket(m: re.Match) -> str:
        open_b, inner, close_b = m.group(1), m.group(2), m.group(3)
        if inner.startswith('"') and inner.endswith('"'):
            return f"{open_b}{inner}{close_b}"
        if inner.startswith("(") and inner.endswith(")"):
            inner_sub = inner[1:-1].strip()
            if not (inner_sub.startswith('"') and inner_sub.endswith('"')):
                inner_sub = f'"{inner_sub}"'
            return f"{open_b}({inner_sub}){close_b}"
        clean_inner = inner.replace('"', '\\"')
        return f'{open_b}"{clean_inner}"{close_b}'

    quote_pattern = re.compile(r'([\[\({])(?!")([^\]\)\}"]*?[ \t:;,?!/\-*][^\]\)\}"]*?)(?<!\")([\]\)}])')
    fixed = quote_pattern.sub(_quote_bracket, fixed)

    return fixed


def repair_mermaid_content(content: str) -> str:
    """
    Scans content for Mermaid code blocks (closed or unclosed),
    repairs each diagram, and ensures all code fences are properly closed.
    """
    if not content or "```mermaid" not in content.lower():
        return content

    # Find closed blocks
    results = []
    closed_spans = []
    for m in re.finditer(r"```mermaid\s*\n?(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE):
        results.append((m.start(), m.end(), m.group(1)))
        closed_spans.append((m.start(), m.end()))

    # Find unclosed block at end of content
    unclosed_match = re.search(r"```mermaid\s*\n?(.*?)$", content, flags=re.DOTALL | re.IGNORECASE)
    if unclosed_match:
        u_start = unclosed_match.start()
        is_already_closed = any(c_s <= u_start < c_e for c_s, c_e in closed_spans)
        if not is_already_closed:
            results.append((u_start, len(content), unclosed_match.group(1)))

    if not results:
        return content

    # Replace from back to front
    results.sort(key=lambda x: x[0], reverse=True)
    for start, end, block_code in results:
        fixed_code = autofix_mermaid_code(block_code)
        new_block = f"```mermaid\n{fixed_code}\n```"
        content = content[:start] + new_block + content[end:]

    return content


def render_message_with_mermaid(content: str, key_suffix: str = ""):
    """Parses a message, repairs Mermaid diagrams, and renders them cleanly."""
    if not content:
        return

    # 1. Pre-process content to repair all Mermaid blocks and close unclosed fences
    repaired_content = repair_mermaid_content(content)

    # 2. Split content into Markdown and Mermaid segments
    parts = re.split(
        r"(```mermaid\s*\n?.*?\n?```)", repaired_content, flags=re.DOTALL | re.IGNORECASE
    )

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        if part.lower().startswith("```mermaid"):
            # Extract mermaid code by stripping fences
            mermaid_code = re.sub(r"^```mermaid\s*\n?", "", part, flags=re.IGNORECASE)
            mermaid_code = re.sub(r"\n?```$", "", mermaid_code).strip()

            if mermaid_code:
                # Ensure the extracted code is auto-fixed
                fixed_mermaid_code = autofix_mermaid_code(mermaid_code)
                unique_id = f"mermaid-{uuid.uuid4().hex[:8]}-{i}"
                escaped_code = html.escape(fixed_mermaid_code)

                # Dynamic height estimation based on lines
                line_count = len(fixed_mermaid_code.splitlines())
                calc_height = min(750, max(220, line_count * 32 + 80))

                mermaid_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: {{ htmlLabels: true, curve: 'basis' }}
        }});

        async function renderDiagram() {{
            const rawCode = document.getElementById('code-{unique_id}').textContent;
            const container = document.getElementById('container-{unique_id}');
            try {{
                const {{ svg }} = await mermaid.render('svg-{unique_id}', rawCode);
                container.innerHTML = svg;
            }} catch (err) {{
                console.warn('Mermaid render error:', err);
                container.innerHTML = `
                    <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 12px; font-family: monospace; color: #cbd5e1; font-size: 12px; text-align: left;">
                        <div style="color: #60a5fa; font-weight: 600; margin-bottom: 6px;">📊 Diagram Structure</div>
                        <pre style="margin: 0; overflow-x: auto; white-space: pre-wrap;">${{rawCode.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</pre>
                    </div>
                `;
            }}
        }}
        window.addEventListener('DOMContentLoaded', renderDiagram);
    </script>
    <style>
        body {{
            margin: 0;
            padding: 8px;
            background: transparent;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        #container-{unique_id} {{
            width: 100%;
            display: flex;
            justify-content: center;
            overflow-x: auto;
        }}
        svg {{
            max-width: 100% !important;
            height: auto !important;
        }}
    </style>
</head>
<body>
    <div id="container-{unique_id}">
        <div style="color: #94a3b8; font-size: 12px;">⏳ Rendering diagram...</div>
    </div>
    <div id="code-{unique_id}" style="display:none;">{escaped_code}</div>
</body>
</html>"""

                try:
                    components.html(mermaid_html, height=calc_height, scrolling=True)
                except Exception as e:
                    logger.error(f"Failed to render Mermaid component: {e}")
                    st.code(fixed_mermaid_code, language="mermaid")
        else:
            # Render regular markdown
            st.markdown(part)
