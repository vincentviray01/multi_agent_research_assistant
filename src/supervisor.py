from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # For short-term memory (thread-level state persistence)
from langgraph.store.memory import (
    InMemoryStore,
)  # For long-term memory (storing user preferences)
from langgraph_supervisor import (
    create_supervisor,
)

from agents import (
    get_python_coding_agent,
    get_python_language_agent,
    get_python_library_agent,
)
from prompts import generate_python_documentation_supervisor_prompt
from state import State
from utils import show_graph


def get_python_documentation_supervisor(plot_graph=False):
    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )

    python_library_agent = get_python_library_agent()
    python_language_agent = get_python_language_agent()
    python_coding_agent = get_python_coding_agent()

    supervisor_prompt = generate_python_documentation_supervisor_prompt()

    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    # Create the supervisor workflow using the `create_supervisor` utility.
    # This function dynamically sets up the graph to route between the provided agents.
    supervisor_prebuilt_workflow = create_supervisor(
        agents=[
            python_language_agent,
            python_library_agent,
            python_coding_agent,
        ],  # List of sub-agents the supervisor can route to
        output_mode="last_message",  # Specifies that the supervisor should output only the last message from the routed agent.
        # Alternative is "full_history" to get all messages from the sub-agent.
        model=llm,  # The LLM to act as the supervisor (for routing decisions).
        prompt=(
            supervisor_prompt
        ),  # The system prompt guiding the supervisor's behavior.
        state_schema=State,  # The shared state schema for the entire multi-agent graph.
    )

    # Compile the supervisor workflow into a runnable object.
    # This makes it ready for invocation and integrates it with our memory systems.
    supervisor_prebuilt = supervisor_prebuilt_workflow.compile(
        name="supervisor", checkpointer=checkpointer, store=in_memory_store
    )

    # Display a visualization of the compiled supervisor graph.
    # Notice how the supervisor acts as the central hub, directing traffic to its sub-agents.
    if plot_graph:
        show_graph(supervisor_prebuilt)

    return supervisor_prebuilt
