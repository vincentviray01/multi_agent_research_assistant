import uuid  # Module for generating unique identifiers

from langchain_core.messages import (
    HumanMessage,
)  # Message types for conversation history
from langgraph.types import Command

from agents import get_invoice_information_agent, get_music_catalog_agent
from state_graph import get_python_documentation_state_graph
from supervisor import get_python_documentation_supervisor


def test_music_catalog_subagent():
    music_catalog_subagent = get_music_catalog_agent()
    thread_id = uuid.uuid4()
    question = "I like the Rolling Stones. What songs do you recommend by them or by other artists that I might like?"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = music_catalog_subagent.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )

    for message in final_state["messages"]:
        message.pretty_print()


def test_invoice_information_subagent():
    invoice_information_subagent = get_invoice_information_agent()
    thread_id = uuid.uuid4()
    question = "My customer id is 1. What was my most recent invoice, and who was the employee that helped me with it?"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    final_state = invoice_information_subagent.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )

    # Print the conversation history from the result for verification.
    for message in final_state["messages"]:
        message.pretty_print()


def test_supervisor():
    supervisor_prebuilt = get_python_documentation_supervisor()
    thread_id = uuid.uuid4()  # Generate a fresh thread ID for this conversation.

    question = "My customer ID is 1. How much was my most recent purchase? What albums do you have by U2?"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    result = supervisor_prebuilt.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )

    # Print the messages from the resulting state to see the conversation flow and final answer.
    for message in result["messages"]:
        message.pretty_print()


def test_human_in_the_loop():
    multi_agent_verify_graph = get_python_documentation_state_graph(plot_graph=True)
    thread_id = uuid.uuid4()  # Generate a new unique thread ID.

    # Initial question without providing customer ID.
    question = "How much was my most recent purchase?"

    # Configuration for the graph invocation.
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph. This first invocation should hit the `human_input` node and interrupt.
    result = multi_agent_verify_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )

    # Print messages to observe the agent asking for the customer ID.
    for message in result["messages"]:
        message.pretty_print()

    # Resume from the interrupt, providing the phone number for verification
    question = "My phone number is +55 (12) 3923-5555."
    result = multi_agent_verify_graph.invoke(Command(resume=question), config=config)
    for message in result["messages"]:
        message.pretty_print()

        # Follow-up question in the same thread (using the same `thread_id`).
    question = "What albums do you have by the Rolling Stones?"

    # Invoke the graph again. Since the `customer_id` is already in the state,
    # the verification step will be skipped, and the query will directly go to the supervisor.
    result = multi_agent_verify_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )

    # Print the results. You should see the music catalog sub-agent's response directly.
    for message in result["messages"]:
        message.pretty_print()


def test_final():
    multi_agent_final_graph = get_python_documentation_state_graph(plot_graph=False)
    thread_id = (
        uuid.uuid4()
    )  # Generate a fresh unique thread ID for this demonstration.

    # A comprehensive question that includes customer ID, invoice query, and music preference.
    question = "My phone number is +55 (12) 3923-5555. How much was my most recent purchase? What albums do you have by the Rolling Stones?"

    # Configuration for the graph invocation.
    # Note: The user_id is passed as a configurable parameter, although in this specific example,
    # the customer_id is extracted dynamically by the verify_info node.
    # For real-world use, ensure consistent handling of user identifiers.
    config = {"configurable": {"thread_id": thread_id, "user_id": "1"}}

    # Invoke the final multi-agent graph.
    # This will run through verification, memory loading, supervisor routing (to invoice then music),
    # and finally memory saving.
    result = multi_agent_final_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
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
    if memory_data and "memory" in memory_data.value:
        saved_music_preferences = memory_data.value.get("memory").music_preferences
    else:
        saved_music_preferences = []  # Default to empty list if no preferences found.

    print(
        f"Saved Music Preferences for Customer ID {user_id}: {saved_music_preferences}"
    )


def test_python_documentation_agent():
    multi_agent_final_graph = get_python_documentation_state_graph()
    thread_id = uuid.uuid4()

    # question = "What is the syntax for the os.mkdir command? How does the Python interpreter work?"
    question = "Write me Python code to list all folders in my current directory."

    config = {"configurable": {"thread_id": thread_id, "user_id": "1"}}

    # Invoke the final multi-agent graph.
    # This will run through verification, memory loading, supervisor routing (to invoice then music),
    # and finally memory saving.
    result = multi_agent_final_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
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
    if memory_data and "memory" in memory_data.value:
        saved_music_preferences = memory_data.value.get("memory").music_preferences
    else:
        saved_music_preferences = []  # Default to empty list if no preferences found.

    print(
        f"Saved Music Preferences for Customer ID {user_id}: {saved_music_preferences}"
    )
