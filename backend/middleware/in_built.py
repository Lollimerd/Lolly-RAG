import os
import logging
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    PIIMiddleware,
)

from setup.init_config import summarizer, answer_LLM

logger = logging.getLogger(__name__)

# --- In-Built Agent Middlewares ---

# 1. Summarization Middleware: compresses older conversation history when token limits approach
summarize = SummarizationMiddleware(
    model=summarizer(), trigger=("tokens", 2000), keep=("messages", 20)
)

# 2. Context Editing Middleware: keeps recent tool uses while clearing stale ones
clear_tool_uses = ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit(
            trigger=100000,
            keep=3,
        ),
    ],
)

# 3. Tool Call Limit Middleware: prevents infinite tool invocation loops (run-level limit)
tool_limit = ToolCallLimitMiddleware(run_limit=2)

# 4. Tool Retry Middleware: automatically retries transient tool failures with exponential backoff
tool_retry = ToolRetryMiddleware(max_retries=2, backoff_factor=2.0)

# 5. Specialized SubAgent specification for Mermaid diagram auto-correction & validation
mermaid_subagent = {
    "name": "mermaid_validator",
    "description": (
        "Validates and auto-corrects malformed Mermaid diagram syntax. "
        "Fixes unquoted shape labels, ensures alphanumeric node identifiers, "
        "balances subgraphs with matching 'end' statements, and ensures valid diagram headers."
    ),
    "system_prompt": (
        "You are an expert Mermaid diagram syntax repair engine. "
        "When given a malformed or broken Mermaid diagram, "
        "you fix all syntax errors, enclose all node labels in double quotes, "
        "ensure all node IDs are alphanumeric/underscores only ([a-zA-Z0-9_]+), "
        "ensure all subgraphs have matching 'end', "
        "ensure spaces exist around all arrows (e.g. 'A --> B'), "
        "and output ONLY the 100% valid Mermaid diagram syntax."
    ),
    "model": answer_LLM(),
    "tools": [],
}
