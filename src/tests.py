import uuid  # Module for generating unique identifiers

from langchain_core.messages import (
    HumanMessage,
)

from state_graph import get_python_documentation_state_graph


def test_python_documentation_agent():
    multi_agent_final_graph = get_python_documentation_state_graph()
    thread_id = uuid.uuid4()

    config = {"configurable": {"thread_id": thread_id}, "user_id": "1"}

    print("--- Python Documentation Agent (Iterative Chat) ---")
    print("Type 'quit', 'exit', or 'q' to stop the chat.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        result = multi_agent_final_graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )

        for message in result["messages"]:
            message.pretty_print()

    user_id = "1"
    namespace = (
        "memory_profile",
        user_id,
    )

    memory_data = multi_agent_final_graph.store.get(namespace, "user_memory")

    if memory_data and "code" in memory_data.value:
        code = memory_data.value.get("code")
    else:
        code = ""

    print(f"Current code for User ID {user_id}: {code}")
