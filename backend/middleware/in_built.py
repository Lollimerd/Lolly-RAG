from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from setup.init_config import summarizer

summarize = SummarizationMiddleware(model=summarizer(), trigger=("tokens", 2000), keep=("messages", 20))
clear_tool_uses = ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100000, keep=3)])
tool_limit = ToolCallLimitMiddleware(run_limit=3)
tool_retry = ToolRetryMiddleware(max_retries=3, backoff_factor=2.0)