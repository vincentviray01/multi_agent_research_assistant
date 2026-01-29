import uuid  # Module for generating unique identifiers

from langchain_core.messages import (
    HumanMessage,
)  # Message types for conversation history

from state_graph import get_python_documentation_state_graph


def test_python_documentation_agent():
    multi_agent_final_graph = get_python_documentation_state_graph()
    thread_id = uuid.uuid4()

    # # Test memory
    # user_id = (
    #     "1"  # The customer ID we expect to be associated with the phone number used.
    # )
    # namespace = (
    #     "memory_profile",
    #     user_id,
    # )  # The namespace used to store this user's memory.

    # Retrieve the user's memory profile from the `in_memory_store`.
    # `.value` retrieves the actual data stored, which should be a dictionary containing the UserProfile instance.
    # memory_data = multi_agent_final_graph.store.get(namespace, "user_memory")

    # # Check if memory_data exists and has a 'memory' key (which holds the UserProfile object).
    # if memory_data and "code" in memory_data.value:
    #     code = memory_data.value.get("code")
    # else:
    #     code = ""  # Default to empty list if no preferences found.
    #
    # print(f"Current code for User ID {user_id}: {code}")

    # question = "What is the syntax for the os.mkdir command? How does the Python interpreter work?"
    # question = "My user ID is 1. Write me Python code to list all folders in my current directory."

    config = {"configurable": {"thread_id": thread_id}, "user_id": "1"}

    print("--- Python Documentation Agent (Iterative Chat) ---")
    print("Type 'quit', 'exit', or 'q' to stop the chat.\n")

    while True:
        # user_input = "My user ID is 1. Write me Python code to list all folders in my current directory."
        user_input = input("User: ")
        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Invoke the final multi-agent graph.
        # This will run through verification, memory loading, supervisor routing (to invoice then music),
        # and finally memory saving.
        result = multi_agent_final_graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )

        # Print all messages in the final state to observe the complete interaction flow.
        for message in result["messages"]:
            message.pretty_print()

    # Test memory
    user_id = (
        "1"  # The customer ID we expect to be associated with the phone number used.
    )
    namespace = (
        "memory_profile",
        user_id,
    )  # The namespace used to store this user's memory.

    # Retrieve the user's memory profile from the `in_memory_store`.
    # `.value` retrieves the actual data stored, which should be a dictionary containing the UserProfile instance.
    memory_data = multi_agent_final_graph.store.get(namespace, "user_memory")

    # Check if memory_data exists and has a 'memory' key (which holds the UserProfile object).
    if memory_data and "code" in memory_data.value:
        code = memory_data.value.get("code")
    else:
        code = ""  # Default to empty list if no preferences found.

    print(f"Current code for User ID {user_id}: {code}")
