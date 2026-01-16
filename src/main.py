import rootpath
from dotenv import load_dotenv  # Import function to load environment variables

from tests import test_python_documentation_agent

if __name__ == "__main__":
    project_root = rootpath.detect()
    load_dotenv(dotenv_path=project_root + "/.env", override=True)

    test_python_documentation_agent()
