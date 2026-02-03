from langchain_core.runnables import RunnableConfig

from state import State


# Define a conditional edge function named `should_continue`.
# This function determines the next step in the graph based on the LLM's response.
def should_continue(state: State, config: RunnableConfig):
    messages = state["messages"]
    last_message = messages[-1]

    # LLMs generate `tool_calls` when they decide to use a function.
    if not last_message.tool_calls:
        # If there are no tool calls, it means the LLM has generated a final answer.
        # In this case, the sub-agent's work is done, so we return "end" to signal completion.
        return "end"
    else:
        return "continue"


# Define the conditional edge function for `verify_info`.
def should_interrupt(state: State, config: RunnableConfig):
    if state.get("user_id") is not None:
        return "continue"
    else:
        return "interrupt"
