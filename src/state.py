from typing import Annotated  # For type hinting lists and adding annotations

from langgraph.graph.message import (
    AnyMessage,
    add_messages,
)  # For managing messages in the graph state
from typing_extensions import TypedDict  # For defining dictionaries with type hints


class State(TypedDict):
    """Represents the state of our LangGraph agent."""

    # messages: A list of messages that form the conversation history.
    # Annotated with `add_messages` to ensure new messages are appended rather than overwritten.
    messages: Annotated[list[AnyMessage], add_messages]

    user_id: str
    # loaded_memory: Stores information loaded from the long-term memory store,
    # typically user preferences or historical context.
    loaded_memory: str

    # remaining_steps: Used by LangGraph to track the number of allowed steps
    # to prevent infinite loops in cyclic graphs.
    remaining_steps: int
