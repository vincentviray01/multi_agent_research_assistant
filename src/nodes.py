from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import (
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState
from langgraph.store.base import (
    BaseStore,
)
from langgraph.types import (
    interrupt,
)

from prompts import (
    generate_create_memory_prompt,
    generate_structured_system_prompt,
)
from schemas import UserInput, UserProfile
from state import State
from utils import format_user_memory


def verify_info(state: State, config: RunnableConfig):
    """Verify the user's current session by parsing their input and matching it with the database."""

    if state.get("user_id") is None:
        user_input = state["messages"][-1]

        llm = ChatOllama(
            model="qwen3",
            temperature=0,
        )

        structured_llm = llm.with_structured_output(schema=UserInput)
        structured_system_prompt = generate_structured_system_prompt()

        parsed_info = structured_llm.invoke(
            [SystemMessage(content=structured_system_prompt)] + [user_input]
        )

        user_id = parsed_info.identifier
        print("UserID", user_id)

        while user_id == "":
            print("Enter your user_id: ")
            user_input = interrupt("User: ")
            result = multi_agent_final_graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config=config
            )

            user_id = parsed_info.identifier
            response = llm.invoke(
                [SystemMessage(content=structured_system_prompt)] + state["messages"]
            )

        intent_message = SystemMessage(
            content=f"Thank you for providing your information! I was able to verify your account with user id {user_id}."
        )
        return {"user_id": user_id, "messages": [intent_message]}

    else:
        pass


def human_input(state: State, config: RunnableConfig):
    """No-op node that should be interrupted on"""
    user_input = interrupt("Please provide input.")

    return {"messages": [user_input]}


def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """Loads generated code from coding agent, if available."""

    user_id = state["user_id"]
    namespace = (
        "memory_profile",
        user_id,
    )

    existing_memory = store.get(namespace, "user_memory")

    formatted_memory = ""  # Initialize formatted memory as empty.

    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    return {"loaded_memory": formatted_memory}


def create_memory(state: State, config: RunnableConfig, store: BaseStore):
    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )

    user_id = str(state["user_id"])
    namespace = (
        "memory_profile",
        user_id,
    )

    existing_memory = store.get(namespace, "user_memory")

    create_memory_prompt = generate_create_memory_prompt()
    formatted_memory = ""
    if existing_memory and existing_memory.value:
        existing_memory_dict = existing_memory.value
        formatted_memory = f"Code: {existing_memory_dict.get('code')}"

    formatted_system_message = SystemMessage(
        content=create_memory_prompt.format(
            conversation=state["messages"], memory_profile=formatted_memory
        )
    )

    updated_memory = llm.with_structured_output(UserProfile).invoke(
        [formatted_system_message]
    )

    key = "user_memory"

    store.put(namespace, key, {"code": updated_memory})


def python_language_rag():
    EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=EMBEDDING_MODEL
    )
    retriever = vectorstore.as_retriever()

    @tool
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

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=EMBEDDING_MODEL
    )
    retriever = vectorstore.as_retriever()

    @tool
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
