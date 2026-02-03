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

    multi_agent_final = StateGraph(State)

    multi_agent_final.add_node("verify_info", verify_info)
    multi_agent_final.add_node("supervisor", supervisor_prebuilt)
    multi_agent_final.add_node("create_memory", create_memory)

    multi_agent_final.add_edge(START, "verify_info")
    multi_agent_final.add_edge("verify_info", "supervisor")

    multi_agent_final.add_edge("supervisor", "create_memory")
    multi_agent_final.add_edge("create_memory", END)

    multi_agent_final_graph = multi_agent_final.compile(
        name="multi_agent_verify", checkpointer=checkpointer, store=in_memory_store
    )

    if plot_graph:
        show_graph(multi_agent_final_graph)

    return multi_agent_final_graph
