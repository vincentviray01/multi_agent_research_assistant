from typing import Annotated  # For type hinting lists and adding annotations

from langgraph.graph.message import (
    AnyMessage,
    add_messages,
)
from typing_extensions import TypedDict  # For defining dictionaries with type hints


class State(TypedDict):
    """Represents the state of our LangGraph agent."""

    messages: Annotated[list[AnyMessage], add_messages]

    user_id: str
    loaded_memory: str

    remaining_steps: int
