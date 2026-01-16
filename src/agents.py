import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # For short-term memory (thread-level state persistence)
from langgraph.prebuilt import (
    create_react_agent,
)
from langgraph.store.memory import (
    InMemoryStore,
)  # For long-term memory (storing user preferences)

from prompts import (
    generate_invoice_subagent_prompt,
    generate_music_assistant_prompt,
    generate_python_coding_prompt,
    generate_python_language_documentation_prompt,
    generate_python_library_documentation_prompt,
)
from state import State
from tools import (
    check_for_songs,
    get_albums_by_artist,
    get_employee_by_invoice_and_customer,
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
    get_songs_by_genre,
    get_tracks_by_artist,
)


def get_invoice_information_agent():
    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )
    invoice_tools = [
        get_invoices_by_customer_sorted_by_date,
        get_invoices_sorted_by_unit_price,
        get_employee_by_invoice_and_customer,
    ]

    invoice_subagent_prompt = generate_invoice_subagent_prompt()

    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    invoice_information_subagent = create_react_agent(
        llm,  # The language model to use for reasoning
        tools=invoice_tools,  # The list of tools available to this agent
        name="invoice_information_subagent",  # A unique name for this agent within the graph
        prompt=invoice_subagent_prompt,  # The system prompt for this agent's persona and instructions
        state_schema=State,  # The shared state schema for the graph
        checkpointer=checkpointer,  # The checkpointer for short-term (thread-level) memory
        store=in_memory_store,  # The in-memory store for long-term user data
    )

    return invoice_information_subagent


def get_music_catalog_agent():
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

    music_assistant_prompt = generate_music_assistant_prompt()

    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    music_catalog_subagent = create_react_agent(
        llm,  # The language model to use for reasoning
        tools=music_tools,  # The list of tools available to this agent
        name="music_catalog_subagent",  # A unique name for this agent within the graph
        prompt=music_assistant_prompt,  # The system prompt for this agent's persona and instructions
        state_schema=State,  # The shared state schema for the graph
        checkpointer=checkpointer,  # The checkpointer for short-term (thread-level) memory
        store=in_memory_store,  # The in-memory store for long-term user data
    )

    return music_catalog_subagent


def get_python_language_agent():
    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )
    EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text:v1.5")

    PYTHON_LANGUAGE_DOCS = [
        "https://docs.python.org/3/reference/index.html",
        "https://docs.python.org/3/reference/introduction.html",
        "https://docs.python.org/3/reference/lexical_analysis.html",
        "https://docs.python.org/3/reference/datamodel.html",
        "https://docs.python.org/3/reference/executionmodel.html",
        "https://docs.python.org/3/reference/import.html",
        "https://docs.python.org/3/reference/expressions.html",
        "https://docs.python.org/3/reference/simple_stmts.html",
        "https://docs.python.org/3/reference/compound_stmts.html",
        "https://docs.python.org/3/reference/toplevel_components.html",
        "https://docs.python.org/3/reference/grammar.html",
    ]

    docs = [WebBaseLoader(url).load() for url in PYTHON_LANGUAGE_DOCS]

    # print(docs[0][0].page_content.strip()[:1000])

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # print(doc_splits[0].page_content.strip())

    # vector_store = InMemoryVectorStore.from_documents(
    #     documents=doc_splits, embedding=EMBEDDING_MODEL
    # )

    embedding_dim = len(EMBEDDING_MODEL.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=EMBEDDING_MODEL,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    retriever = vector_store.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="python_language_search",
        description="Searches for documentation and code examples for the Python library. "
        "Use this tool when you need to answer technical questions about library functions.",
    )

    python_library_prompt = generate_python_language_documentation_prompt()

    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    python_language_subagent = create_react_agent(
        llm,  # The language model to use for reasoning
        tools=[retriever_tool],  # The list of tools available to this agent
        name="python_language_agent",  # A unique name for this agent within the graph
        prompt=python_library_prompt,  # The system prompt for this agent's persona and instructions
        state_schema=State,  # The shared state schema for the graph
        checkpointer=checkpointer,  # The checkpointer for short-term (thread-level) memory
        store=in_memory_store,  # The in-memory store for long-term user data
    )

    return python_language_subagent


def get_python_library_agent():
    llm = ChatOllama(
        model="qwen3",
        temperature=0,
    )
    EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text:v1.5")

    PYTHON_LIBRARY_DOCS = [
        "https://docs.python.org/3/library/index.html",
        "https://docs.python.org/3/library/intro.html",
        "https://docs.python.org/3/library/functions.html",
        "https://docs.python.org/3/library/constants.html",
        "https://docs.python.org/3/library/stdtypes.html",
        "https://docs.python.org/3/library/exceptions.html",
        "https://docs.python.org/3/library/text.html",
        "https://docs.python.org/3/library/binary.html",
        "https://docs.python.org/3/library/datatypes.html",
        "https://docs.python.org/3/library/numeric.html",
        "https://docs.python.org/3/library/functional.html",
        "https://docs.python.org/3/library/filesys.html",
        "https://docs.python.org/3/library/persistence.html",
        "https://docs.python.org/3/library/archiving.html",
        "https://docs.python.org/3/library/fileformats.html",
        "https://docs.python.org/3/library/crypto.html",
        "https://docs.python.org/3/library/allos.html",
        "https://docs.python.org/3/library/cmdlinelibs.html",
        "https://docs.python.org/3/library/concurrency.html",
        "https://docs.python.org/3/library/ipc.html",
        "https://docs.python.org/3/library/netdata.html",
        "https://docs.python.org/3/library/markup.html",
        "https://docs.python.org/3/library/internet.html",
        "https://docs.python.org/3/library/mm.html",
        "https://docs.python.org/3/library/i18n.html",
        "https://docs.python.org/3/library/tk.html",
        "https://docs.python.org/3/library/development.html",
        "https://docs.python.org/3/library/debug.html",
        "https://docs.python.org/3/library/distribution.html",
        "https://docs.python.org/3/library/python.html",
        "https://docs.python.org/3/library/custominterp.html",
        "https://docs.python.org/3/library/modules.html",
        "https://docs.python.org/3/library/language.html",
        "https://docs.python.org/3/library/windows.html",
        "https://docs.python.org/3/library/unix.html",
        "https://docs.python.org/3/library/cmdline.html",
        "https://docs.python.org/3/library/ast.html",
        "https://docs.python.org/3/library/asyncio.html",
        "https://docs.python.org/3/library/base64.html",
        "https://docs.python.org/3/library/calendar.html",
        "https://docs.python.org/3/library/code.html",
        "https://docs.python.org/3/library/compileall.html",
        "https://docs.python.org/3/library/profile.html",
        "https://docs.python.org/3/library/dis.html",
        "https://docs.python.org/3/library/doctest.html",
        "https://docs.python.org/3/library/ensurepip.html",
        "https://docs.python.org/3/library/filecmp.html",
        "https://docs.python.org/3/library/fileinput.html",
        "https://docs.python.org/3/library/ftplib.html",
        "https://docs.python.org/3/library/gzip.html",
        "https://docs.python.org/3/library/http.server.html",
        "https://docs.python.org/3/library/idle.html",
        "https://docs.python.org/3/library/inspect.html",
        "https://docs.python.org/3/library/json.html",
        "https://docs.python.org/3/library/mimetypes.html",
        "https://docs.python.org/3/library/pdb.html",
        "https://docs.python.org/3/library/pickle.html",
        "https://docs.python.org/3/library/pickletools.html",
        "https://docs.python.org/3/library/platform.html",
        "https://docs.python.org/3/library/poplib.html",
        "https://docs.python.org/3/library/profile.html",
        "https://docs.python.org/3/library/profile.html",
        "https://docs.python.org/3/library/py_compile.html",
        "https://docs.python.org/3/library/pyclbr.html",
        "https://docs.python.org/3/library/pydoc.html",
        "https://docs.python.org/3/library/quopri.html",
        "https://docs.python.org/3/library/random.html",
        "https://docs.python.org/3/library/runpy.html",
        "https://docs.python.org/3/library/site.html",
        "https://docs.python.org/3/library/sqlite3.html",
        "https://docs.python.org/3/library/symtable.html",
        "https://docs.python.org/3/library/sysconfig.html",
        "https://docs.python.org/3/library/tabnanny.html",
        "https://docs.python.org/3/library/tarfile.html",
        "https://docs.python.org/3/library/timeit.html",
        "https://docs.python.org/3/library/tokenize.html",
        "https://docs.python.org/3/library/trace.html",
        "https://docs.python.org/3/library/turtle.html",
        "https://docs.python.org/3/library/unittest.html",
        "https://docs.python.org/3/library/uuid.html",
        "https://docs.python.org/3/library/venv.html",
        "https://docs.python.org/3/library/webbrowser.html",
        "https://docs.python.org/3/library/zipapp.html",
        "https://docs.python.org/3/library/zipfile.html"
        "https://docs.python.org/3/library/superseded.html",
        "https://docs.python.org/3/library/removed.html",
        "https://docs.python.org/3/library/aifc.html",
        "https://docs.python.org/3/library/asynchat.html",
        "https://docs.python.org/3/library/asyncore.html",
        "https://docs.python.org/3/library/audioop.html",
        "https://docs.python.org/3/library/cgi.html",
        "https://docs.python.org/3/library/cgitb.html",
        "https://docs.python.org/3/library/chunk.html",
        "https://docs.python.org/3/library/crypt.html",
        "https://docs.python.org/3/library/distutils.html",
        "https://docs.python.org/3/library/imghdr.html",
        "https://docs.python.org/3/library/imp.html",
        "https://docs.python.org/3/library/mailcap.html",
        "https://docs.python.org/3/library/msilib.html",
        "https://docs.python.org/3/library/nis.html",
        "https://docs.python.org/3/library/nntplib.html",
        "https://docs.python.org/3/library/ossaudiodev.html",
        "https://docs.python.org/3/library/pipes.html",
        "https://docs.python.org/3/library/smtpd.html",
        "https://docs.python.org/3/library/sndhdr.html",
        "https://docs.python.org/3/library/spwd.html",
        "https://docs.python.org/3/library/sunau.html",
        "https://docs.python.org/3/library/telnetlib.html",
        "https://docs.python.org/3/library/uu.html",
        "https://docs.python.org/3/library/xdrlib.html",
    ]

    docs = [WebBaseLoader(url).load() for url in PYTHON_LIBRARY_DOCS]

    # print(docs[0][0].page_content.strip()[:1000])

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # print(doc_splits[0].page_content.strip())

    # vector_store = InMemoryVectorStore.from_documents(
    #     documents=doc_splits, embedding=EMBEDDING_MODEL
    # )

    embedding_dim = len(EMBEDDING_MODEL.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=EMBEDDING_MODEL,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    retriever = vector_store.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="python_library_search",
        description="Searches for documentation and code examples for the Python library. "
        "Use this tool when you need to answer technical questions about library functions.",
    )

    python_library_prompt = generate_python_library_documentation_prompt()

    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    python_library_subagent = create_react_agent(
        llm,  # The language model to use for reasoning
        tools=[retriever_tool],  # The list of tools available to this agent
        name="python_library_agent",  # A unique name for this agent within the graph
        prompt=python_library_prompt,  # The system prompt for this agent's persona and instructions
        state_schema=State,  # The shared state schema for the graph
        checkpointer=checkpointer,  # The checkpointer for short-term (thread-level) memory
        store=in_memory_store,  # The in-memory store for long-term user data
    )

    return python_library_subagent


def get_python_coding_agent():
    llm = ChatOllama(
        model="vndr/pythoncoder",
        temperature=0,
    )

    python_coding_prompt = generate_python_coding_prompt()

    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    python_language_subagent = create_react_agent(
        llm,  # The language model to use for reasoning
        tools=[],  # The list of tools available to this agent
        name="python_coding_agent",  # A unique name for this agent within the graph
        prompt=python_coding_prompt,  # The system prompt for this agent's persona and instructions
        state_schema=State,  # The shared state schema for the graph
        checkpointer=checkpointer,  # The checkpointer for short-term (thread-level) memory
        store=in_memory_store,  # The in-memory store for long-term user data
    )

    return python_language_subagent
