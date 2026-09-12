from langchain.agents.middleware import (
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from setup.init_config import summarizer


summarize = SummarizationMiddleware(
    model=summarizer(), trigger=("tokens", 2000), keep=("messages", 20)
)


clear_tool_uses = ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit(
            trigger=100000,
            keep=3,
        ),
    ],
)


# Tool Call Limit Middleware: prevents infinite tool invocation loops (run-level limit)
tool_limit = ToolCallLimitMiddleware(run_limit=3)

# Tool Retry Middleware: automatically retries transient tool failures with exponential backoff
tool_retry = ToolRetryMiddleware(max_retries=3, backoff_factor=2.0)