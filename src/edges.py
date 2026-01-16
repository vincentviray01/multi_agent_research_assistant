from langchain_core.runnables import RunnableConfig

from state import State


# Define a conditional edge function named `should_continue`.
# This function determines the next step in the graph based on the LLM's response.
def should_continue(state: State, config: RunnableConfig):
    # Get the list of messages from the current state.
    messages = state["messages"]
    # Get the last message, which is the response from the `music_assistant` LLM.
    last_message = messages[-1]

    # Check if the last message contains any tool calls.
    # LLMs generate `tool_calls` when they decide to use a function.
    if not last_message.tool_calls:
        # If there are no tool calls, it means the LLM has generated a final answer.
        # In this case, the sub-agent's work is done, so we return "end" to signal completion.
        return "end"
    # Otherwise, if there are tool calls,
    else:
        # We need to execute the tool(s). So, we return "continue" to route to the tool execution node.
        return "continue"


# Define the conditional edge function for `verify_info`.
# This function checks if a `customer_id` has been successfully set in the state.
def should_interrupt(state: State, config: RunnableConfig):
    # If `customer_id` is present, it means verification was successful or already done, so continue.
    if state.get("customer_id") is not None:
        return "continue"
    # Otherwise, it means customer ID is missing or couldn't be verified, so interrupt for human input.
    else:
        return "interrupt"
