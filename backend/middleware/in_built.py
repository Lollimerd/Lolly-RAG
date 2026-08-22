from langchain.agents.middleware import (
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
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
