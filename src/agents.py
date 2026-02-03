import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import (
    MemorySaver,
)
from langgraph.prebuilt import (
    create_react_agent,
)
from langgraph.store.memory import (
    InMemoryStore,
)

from prompts import (
    generate_python_coding_prompt,
    generate_python_language_documentation_prompt,
    generate_python_library_documentation_prompt,
)
from state import State


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

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

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
        llm,
        tools=[retriever_tool],
        name="python_language_agent",
        prompt=python_library_prompt,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
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

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

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
        llm,
        tools=[retriever_tool],
        name="python_library_agent",
        prompt=python_library_prompt,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
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
        llm,
        tools=[],
        name="python_coding_agent",
        prompt=python_coding_prompt,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
    )

    return python_language_subagent
