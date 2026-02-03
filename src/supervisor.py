from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import (
    MemorySaver,
)
from langgraph.store.memory import (
    InMemoryStore,
)
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

    supervisor_prebuilt_workflow = create_supervisor(
        agents=[
            python_language_agent,
            python_library_agent,
            python_coding_agent,
        ],
        output_mode="last_message",
        model=llm,
        prompt=(supervisor_prompt),
        state_schema=State,
    )

    supervisor_prebuilt = supervisor_prebuilt_workflow.compile(
        name="supervisor", checkpointer=checkpointer, store=in_memory_store
    )

    if plot_graph:
        show_graph(supervisor_prebuilt)

    return supervisor_prebuilt
