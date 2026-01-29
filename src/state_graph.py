from langgraph.checkpoint.memory import (
    MemorySaver,
)  # For short-term memory (thread-level state persistence)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)  # Core LangGraph classes and special node names
from langgraph.store.memory import (
    InMemoryStore,
)  # For long-term memory (storing user preferences)

from nodes import create_memory, verify_info
from state import State
from supervisor import get_python_documentation_supervisor
from utils import show_graph


def get_python_documentation_state_graph(plot_graph=False):
    supervisor_prebuilt = get_python_documentation_supervisor()
    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    # Initialize the final StateGraph for our complete multi-agent system, including memory management.
    multi_agent_final = StateGraph(State)

    # Add all necessary nodes to the graph.
    multi_agent_final.add_node(
        "verify_info", verify_info
    )  # Node for customer verification
    # multi_agent_final.add_node(
    #     "human_input", human_input
    # )  # Node for human-in-the-loop interruption
    # multi_agent_final.add_node(
    #     "load_memory", load_memory
    # )  # Node for loading user long-term memory
    multi_agent_final.add_node(
        "supervisor", supervisor_prebuilt
    )  # Supervisor for routing to sub-agents
    multi_agent_final.add_node(
        "create_memory", create_memory
    )  # Node for saving/updating user long-term memory

    # Define the initial entry point: all interactions start with verification.
    multi_agent_final.add_edge(START, "verify_info")
    multi_agent_final.add_edge("verify_info", "supervisor")

    # Define the conditional routing after `verify_info`.
    # If verification is successful, proceed to load memory. Otherwise, prompt for human input.
    # multi_agent_final.add_conditional_edges(
    #     "verify_info",
    #     should_interrupt,
    #     {
    #         "continue": "load_memory",  # If verified, load user memory
    #         "interrupt": "human_input",  # If not verified, request human input
    #     },
    # )

    # # After `human_input` (resume), loop back to `verify_info` to re-attempt verification.
    # multi_agent_final.add_edge("human_input", "verify_info")
    #
    # # After loading memory, proceed to the supervisor for main query processing.
    # multi_agent_final.add_edge("load_memory", "supervisor")
    #
    # # After the supervisor completes, save/update the user's memory.
    # multi_agent_final.add_edge("supervisor", "create_memory")

    # The graph ends after memory has been updated.
    # multi_agent_final.add_edge("create_memory", END)
    multi_agent_final.add_edge("supervisor", "create_memory")
    multi_agent_final.add_edge("create_memory", END)

    # Compile the entire, sophisticated graph.
    multi_agent_final_graph = multi_agent_final.compile(
        name="multi_agent_verify", checkpointer=checkpointer, store=in_memory_store
    )

    # Display the visualization of the new graph.
    if plot_graph:
        show_graph(multi_agent_final_graph)

    return multi_agent_final_graph
