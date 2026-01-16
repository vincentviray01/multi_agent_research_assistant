import os

from IPython.display import Image
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the embedding model from Together AI
# NOTE: Adjust this import to use a different embedding model

# Initialize the embedding model using Alibaba's GTE ModernBERT base model
# This model is used to convert text into vector representations for similarity search
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


def get_python_library_docs_vectorstore():
    """
    Loads or creates a retriever for Python Library documentation using a persisted Chroma vectorstore.

    This function implements a caching mechanism:
    1. If a vectorstore already exists on disk, it loads and returns it
    2. If no vectorstore exists, it downloads the documentation, creates embeddings,
       stores them in a vectorstore, and persists it to disk for future use

    Returns:
        Retriever: A retriever object for querying the LangGraph documentation using similarity search
    """
    # Check if the vectorstore directory already exists on disk
    # This allows us to skip the expensive document loading and embedding process
    if os.path.exists("python_library_docs_db"):
        print("Loading vectorstore from disk...")
        # Load the existing vectorstore from the persistent directory
        vectorstore = Chroma(
            collection_name="python_library_docs",  # Name of the collection in the vectorstore
            embedding_function=EMBEDDING_MODEL,  # Embedding model for query encoding
            persist_directory="python_library_docs_db",  # Directory where the vectorstore is saved
        )
        # Return a retriever interface for the vectorstore
        # lambda_mult=0 disables the MMR (Maximum Marginal Relevance) diversity filter
        return vectorstore.as_retriever(lambda_mult=0)

    # If vectorstore doesn't exist, create it from scratch
    print("Creating new vectorstore from documentation...")

    # Download and load documents from each URL in LANGGRAPH_DOCS
    # WebBaseLoader fetches the content from web pages and converts them to Document objects
    docs = [WebBaseLoader(url).load() for url in PYTHON_LIBRARY_DOCS]

    # Flatten the list of lists into a single list of documents
    # Each WebBaseLoader.load() returns a list, so we need to flatten the nested structure
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents into smaller chunks for better retrieval performance
    # Smaller chunks allow for more precise similarity matching
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,  # Maximum tokens per chunk (small chunks for precise retrieval)
        chunk_overlap=0,  # No overlap between chunks to avoid redundancy
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma(
        collection_name="python_library_docs",  # Name of the collection in the vectorstore
        embedding_function=EMBEDDING_MODEL,  # Embedding model for query encoding
        persist_directory="python_library_docs_db",  # Directory where the vectorstore is saved
    )

    # Add the split documents to the vectorstore
    # This creates embeddings for each chunk and stores them in the vector database
    vectorstore.add_documents(doc_splits)
    print("Vectorstore created and persisted to disk")

    # Return a retriever interface for the new vectorstore
    return vectorstore.as_retriever(lambda_mult=0)


def get_python_language_docs_vectorstore():
    """
    Loads or creates a retriever for Python Library documentation using a persisted Chroma vectorstore.

    This function implements a caching mechanism:
    1. If a vectorstore already exists on disk, it loads and returns it
    2. If no vectorstore exists, it downloads the documentation, creates embeddings,
       stores them in a vectorstore, and persists it to disk for future use

    Returns:
        Retriever: A retriever object for querying the LangGraph documentation using similarity search
    """
    # Check if the vectorstore directory already exists on disk
    # This allows us to skip the expensive document loading and embedding process
    if os.path.exists("python_language_docs_db"):
        print("Loading vectorstore from disk...")
        # Load the existing vectorstore from the persistent directory
        vectorstore = Chroma(
            collection_name="python_language_docs",  # Name of the collection in the vectorstore
            embedding_function=EMBEDDING_MODEL,  # Embedding model for query encoding
            persist_directory="python_language_docs_db",  # Directory where the vectorstore is saved
        )
        # Return a retriever interface for the vectorstore
        # lambda_mult=0 disables the MMR (Maximum Marginal Relevance) diversity filter
        return vectorstore.as_retriever(lambda_mult=0)

    # If vectorstore doesn't exist, create it from scratch
    print("Creating new vectorstore from documentation...")

    # Download and load documents from each URL in LANGGRAPH_DOCS
    # WebBaseLoader fetches the content from web pages and converts them to Document objects
    docs = [WebBaseLoader(url).load() for url in PYTHON_LIBRARY_DOCS]

    # Flatten the list of lists into a single list of documents
    # Each WebBaseLoader.load() returns a list, so we need to flatten the nested structure
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents into smaller chunks for better retrieval performance
    # Smaller chunks allow for more precise similarity matching
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,  # Maximum tokens per chunk (small chunks for precise retrieval)
        chunk_overlap=0,  # No overlap between chunks to avoid redundancy
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma(
        collection_name="python_language_docs",  # Name of the collection in the vectorstore
        embedding_function=EMBEDDING_MODEL,  # Embedding model for query encoding
        persist_directory="python_language_docs_db",  # Directory where the vectorstore is saved
    )

    # Add the split documents to the vectorstore
    # This creates embeddings for each chunk and stores them in the vector database
    vectorstore.add_documents(doc_splits)
    print("Vectorstore created and persisted to disk")

    # Return a retriever interface for the new vectorstore
    return vectorstore.as_retriever(lambda_mult=0)


def show_graph(graph, xray=False):
    """
    Display a LangGraph mermaid diagram with fallback rendering.

    This function attempts to render a LangGraph as a visual diagram using Mermaid.
    It includes error handling to fall back to an alternative renderer if the default fails.

    Args:
        graph: The LangGraph object that has a get_graph() method for visualization
        xray (bool): Whether to show internal graph details in xray mode

    Returns:
        Image: An IPython Image object containing the rendered graph diagram
    """

    try:
        # Try the default mermaid renderer first (uses mermaid.ink service)
        # This is the fastest option but may fail due to network issues or service unavailability
        return Image(graph.get_graph(xray=xray).draw_mermaid_png(proxies=None))
    except Exception as e:
        # If the default renderer fails, fall back to pyppeteer
        # pyppeteer uses a local headless Chrome instance to render the diagram
        print(f"Default renderer failed ({e}), falling back to pyppeteer...")

        # Apply nest_asyncio to handle async operations in Jupyter environments
        # This is necessary because pyppeteer uses async operations
        import nest_asyncio

        nest_asyncio.apply()

        # Import the MermaidDrawMethod enum for specifying the draw method
        from langchain_core.runnables.graph import MermaidDrawMethod

        # Use pyppeteer as the drawing method (local rendering)
        return Image(
            graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        )


# Helper function to format user memory (music preferences) into a readable string.
def format_user_memory(user_data):
    """Formats music preferences from users, if available."""
    profile = user_data["memory"]  # Access the 'memory' key from the stored dictionary
    result = ""  # Initialize an empty string for the formatted result

    # Check if the profile object has a 'music_preferences' attribute and if it's not empty.
    if hasattr(profile, "music_preferences") and profile.music_preferences:
        # If preferences exist, join them into a comma-separated string.
        result += f"Music Preferences: {', '.join(profile.music_preferences)}"

    return (
        result.strip()
    )  # Return the formatted string, removing any leading/trailing whitespace.
