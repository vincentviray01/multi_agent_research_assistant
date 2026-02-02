from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import (
    SystemMessage,
)  # Message types for conversation history
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.store.base import (
    BaseStore,
)  # Base class for defining custom stores for LangGraph
from langgraph.types import (
    interrupt,
)  # Import the `interrupt` function for pausing graph execution

from prompts import (
    generate_create_memory_prompt,
    generate_structured_system_prompt,
)
from schemas import UserInput, UserProfile
from state import State
from tools import (
    check_for_songs,
    get_albums_by_artist,
    get_songs_by_genre,
    get_tracks_by_artist,
)
from utils import format_user_memory


# This function receives the current `State` and `RunnableConfig`.
def music_assistant(state: State, config: RunnableConfig):
    # Fetch long-term memory (user preferences) from the state.
    # If `loaded_memory` is not present in the state, default to "None".
    memory = "None"
    if "loaded_memory" in state:
        memory = state["loaded_memory"]

    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )
    music_tools = [
        get_albums_by_artist,
        get_tracks_by_artist,
        get_songs_by_genre,
        check_for_songs,
    ]

    llm_with_music_tools = llm.bind_tools(music_tools)
    # Generate the system prompt for the music assistant, injecting the loaded memory.
    music_assistant_prompt = generate_music_assistant_prompt(memory)

    # Invoke the LLM (`llm_with_music_tools`) with the system prompt and the current message history.
    # The LLM will decide whether to call a tool or generate a final response.
    response = llm_with_music_tools.invoke(
        [SystemMessage(music_assistant_prompt)] + state["messages"]
    )

    # Update the state by appending the LLM's response to the `messages` list.
    # The `add_messages` annotation in `State` ensures this is appended correctly.
    return {"messages": [response]}


def get_music_tool_node():
    music_tools = [
        get_albums_by_artist,
        get_tracks_by_artist,
        get_songs_by_genre,
        check_for_songs,
    ]

    music_tool_node = ToolNode(music_tools)

    return music_tool_node


# This node is responsible for verifying the customer's identity based on their input.
def verify_info(state: State, config: RunnableConfig):
    """Verify the user's current session by parsing their input and matching it with the database."""

    # Check if a customer_id is already present in the state.
    # If it is, verification is complete, and the node does nothing (passes).
    if state.get("user_id") is None:
        # Get the most recent user message from the state.
        user_input = state["messages"][-1]

        llm = ChatOllama(
            model="qwen3",
            temperature=0,
        )

        structured_llm = llm.with_structured_output(schema=UserInput)
        structured_system_prompt = generate_structured_system_prompt()

        # Use the structured LLM to parse the user's input for an identifier.
        # It combines the structured system prompt with the user's message.
        parsed_info = structured_llm.invoke(
            [SystemMessage(content=structured_system_prompt)] + [user_input]
        )

        user_id = parsed_info.identifier
        # Extract the identified identifier string.
        print("UserID", user_id)

        while user_id == "":
            print("Enter your user_id: ")
            # user_input = state["messages"][-1]
            user_input = interrupt("User: ")
            result = multi_agent_final_graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config=config
            )

            user_id = parsed_info.identifier
            # Invoke the base LLM with instructions to prompt the user for their identifier or revise it.
            response = llm.invoke(
                [SystemMessage(content=structured_system_prompt)] + state["messages"]
            )
            # Update the state with the LLM's response (the prompt for user input).

        intent_message = SystemMessage(
            content=f"Thank you for providing your information! I was able to verify your account with user id {user_id}."
        )
        return {"user_id": user_id, "messages": [intent_message]}

    else:
        # If `customer_id` is already in state, this node does nothing.
        # This `pass` implies that the graph will simply proceed to the next edge,
        # as defined in the graph compilation.
        pass


# Define the `human_input` node function.
# This node serves as a placeholder to signal that human intervention is required.
def human_input(state: State, config: RunnableConfig):
    """No-op node that should be interrupted on"""
    # `interrupt("Please provide input.")` pauses the graph execution.
    # The string message is passed as a reason for the interrupt.
    # When the graph is resumed, the new input will be stored in `user_input`.
    user_input = interrupt("Please provide input.")

    # The new user input (after resume) is then added to the messages in the state.
    return {"messages": [user_input]}


# Define the `load_memory` node function.
# This node loads a user's long-term memory (music preferences) into the current state.
def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """Loads generated code from coding agent, if available."""

    user_id = state["user_id"]  # Get the current customer ID from the state.
    namespace = (
        "memory_profile",
        user_id,
    )  # Define a namespace for storing user-specific memory.
    # This creates a unique key for each user's profile.

    # Attempt to retrieve existing memory for this user from the `InMemoryStore`.
    existing_memory = store.get(namespace, "user_memory")

    formatted_memory = ""  # Initialize formatted memory as empty.

    # If memory exists and has a value, format it using our helper function.
    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    # Update the `loaded_memory` field in the state with the retrieved and formatted memory.
    return {"loaded_memory": formatted_memory}


# Define the `create_memory` node function.
# This node is responsible for analyzing the conversation and saving/updating user music preferences.
def create_memory(state: State, config: RunnableConfig, store: BaseStore):
    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )

    user_id = str(
        state["user_id"]
    )  # Get the customer ID from the current state (convert to string).
    namespace = (
        "memory_profile",
        user_id,
    )  # Define the namespace for this user's memory profile.

    # Retrieve the existing memory profile for this user from the long-term store.
    existing_memory = store.get(namespace, "user_memory")

    create_memory_prompt = generate_create_memory_prompt()
    formatted_memory = ""  # Initialize formatted memory for the prompt.
    if existing_memory and existing_memory.value:
        existing_memory_dict = (
            existing_memory.value
        )  # Get the dictionary containing the UserProfile instance.
        # Format existing music preferences into a string for the prompt.
        formatted_memory = f"Code: {existing_memory_dict.get('code')}"

    # Create a SystemMessage with the formatted prompt, injecting the full conversation history
    # and the existing memory profile.
    formatted_system_message = SystemMessage(
        content=create_memory_prompt.format(
            conversation=state["messages"], memory_profile=formatted_memory
        )
    )

    # Invoke the LLM with structured output (`UserProfile`) to analyze the conversation
    # and update the memory profile based on new information.
    updated_memory = llm.with_structured_output(UserProfile).invoke(
        [formatted_system_message]
    )

    key = "user_memory"  # Define the key for storing this specific memory object.

    # Store the updated memory profile back into the `InMemoryStore`.
    # We wrap `updated_memory` in a dictionary under the key 'memory' for consistency in access.
    store.put(namespace, key, {"code": updated_memory})


def python_language_rag():
    EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]

    # print(docs[0][0].page_content.strip()[:1000])

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # print(doc_splits[0].page_content.strip())

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=EMBEDDING_MODEL
    )
    retriever = vectorstore.as_retriever()

    @tool
    # the doc string here defines how this node operates
    def retrieve_python_language_documentation(query: str) -> str:
        """Search and return information about Python Language Documentation"""
        docs = retriever.invoke(query)
        print("Docs are", docs)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = retrieve_python_language_documentation

    retriever_tool.invoke({"query": "types of reward hacking"})

    response_model = ChatOllama(
        model="qwen3",
        temperature=0,
    )

    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
        """
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}


def python_library_rag():
    EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]

    # print(docs[0][0].page_content.strip()[:1000])

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # print(doc_splits[0].page_content.strip())

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=EMBEDDING_MODEL
    )
    retriever = vectorstore.as_retriever()

    @tool
    # the doc string here defines how this node operates
    def retrieve_python_language_documentation(query: str) -> str:
        """Search and return information about Python Language Documentation"""
        docs = retriever.invoke(query)
        print("Docs are", docs)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = retrieve_python_language_documentation

    retriever_tool.invoke({"query": "types of reward hacking"})

    response_model = ChatOllama(
        model="qwen3",
        temperature=0,
    )

    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
        """
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}
