import asyncio
import uuid  # For generating unique thread IDs
from typing import Annotated  # For type hinting lists and adding annotations

import rootpath
from dotenv import load_dotenv  # Import function to load environment variables
from langchain_ollama import ChatOllama
from langgraph.types import Command  # For resuming graph execution after an interrupt
from langsmith import (
    Client,
)  # Import the LangSmith Client for dataset and experiment management
from openevals.llm import (
    create_llm_as_judge,
)  # Import the utility to create LLM-as-judge evaluators
from openevals.prompts import (
    CORRECTNESS_PROMPT,
)  # Import a pre-defined prompt for correctness evaluation
from typing_extensions import TypedDict  # For defining dictionaries with type hints

from state_graph import get_state_graph

project_root = rootpath.detect()
load_dotenv(dotenv_path=project_root + "/.env", override=True)


async def run_graph(inputs: dict):
    multi_agent_final_graph = get_state_graph()
    graph = multi_agent_final_graph  # Reference our complete, final multi-agent graph
    """Run graph and track the final response for evaluation."""
    # Creating a unique thread ID for each evaluation run to ensure isolation.
    thread_id = uuid.uuid4()
    # Configuration for the graph invocation. User ID '10' is used here for a specific test scenario.
    configuration = {
        "configurable": {"thread_id": thread_id, "user_id": "10"},
        "recursion_limit": 50,
    }

    # Invoke the graph with the initial user question.
    # This invocation will likely hit the `human_input` node and interrupt if `customer_id` is not present.
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        config=configuration,
    )

    # After the first invocation, if an interrupt occurred, resume it.
    # We explicitly provide a (simulated) customer ID to pass the verification step.
    # The `thread_id` in the config must match the initial invocation to resume the correct state.
    result = await graph.ainvoke(
        Command(resume="My customer ID is 10"),
        config={"configurable": {"thread_id": thread_id, "user_id": "10"}},
    )

    # Return the content of the last message in the conversation as the final response.
    # This is the output that will be evaluated against the dataset's `response`.
    return {"response": result["messages"][-1].content}


llm = ChatOllama(
    model="qwen3",
    temperature=0,
)

# Create an LLM-as-judge evaluator for correctness using the pre-built `CORRECTNESS_PROMPT`.
# `feedback_key="correctness"` sets the name of the score reported in LangSmith.
# `judge=model` specifies which LLM to use for judging.
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=llm,
)

# Custom definition of LLM-as-judge instructions.
# This prompt provides specific guidelines for the LLM acting as a grader, focusing on factual accuracy.
grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

Here is the grade criteria to follow:
(1) Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student response does not contain any conflicting statements.
(3) It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the ground truth response.

Correctness:
True means that the student's response meets all of the criteria.
False means that the student's response does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""


# Define the schema for the LLM-as-judge's output using TypedDict.
# This ensures the grading output is structured with a reasoning and a boolean correctness score.
class Grade(TypedDict):
    """Compare the expected and actual answers and grade the actual answer."""

    reasoning: Annotated[
        str,
        ...,
        "Explain your reasoning for whether the actual response is correct or not.",
    ]
    is_correct: Annotated[
        bool,
        ...,
        "True if the student response is mostly or exactly correct, otherwise False.",
    ]


# Define the custom evaluator function `final_answer_correct`.
# This function takes inputs, outputs (from our `run_graph`), and reference outputs (from the dataset).
async def final_answer_correct(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> bool:
    """Evaluate if the final response is equivalent to reference response."""
    # Construct the user prompt for the grader LLM, combining the question, ground truth, and student response.
    user = f"""QUESTION: {inputs["question"]}
    GROUND TRUTH RESPONSE: {reference_outputs["response"]}
    STUDENT RESPONSE: {outputs["response"]}"""

    # Configure the judge LLM to output structured data according to the `Grade` schema.
    # `method="json_schema"` ensures JSON-based structured output, `strict=True` enforces strict adherence.
    grader_llm = llm.with_structured_output(Grade, method="json_schema", strict=True)

    # Invoke the structured grader LLM with the system instructions and the user prompt.
    # Awaiting the async call as LLM invocations are typically async.
    grade = await grader_llm.ainvoke(
        [
            {"role": "system", "content": grader_instructions},
            {"role": "user", "content": user},
        ]
    )

    # Return the `is_correct` boolean from the grader's output as the evaluation score.
    return grade["is_correct"]


# Run the evaluation job asynchronously using the LangSmith client.
# This will execute `run_graph` for each example in the dataset and apply the specified evaluators.
async def evaluate():
    client = (
        Client()
    )  # Initialize the LangSmith client. This will connect to your LangSmith account.

    # Define a list of example inputs and expected outputs for our dataset.
    # Each dictionary represents a test case with a 'question' (input) and a 'response' (ground truth output).
    examples = [
        {
            "question": "My name is Aaron Mitchell. My number associated with my account is +1 (204) 452-6452. I am trying to find the invoice number for my most recent song purchase. Could you help me with it?",
            "response": "The Invoice ID of your most recent purchase was 342.",
        },
        # {
        #     "question": "I'd like a refund.",
        #     "response": "I need additional information to help you with the refund. Could you please provide your customer identifier so that we can fetch your purchase history?",
        # },
        # {
        #     "question": "Who recorded Wish You Were Here again?",
        #     "response": "Wish You Were Here is an album by Pink Floyd",
        # },
        # {
        #     "question": "What albums do you have by Coldplay?",
        #     "response": "There are no Coldplay albums available in our catalog at the moment.",
        # },
    ]

    dataset_name = (
        "LangGraph 101 Multi-Agent: Final Response"  # Define a name for our dataset.
    )

    # Check if the dataset already exists in LangSmith to avoid recreation.
    if not client.has_dataset(dataset_name=dataset_name):
        # If not, create the dataset.
        dataset = client.create_dataset(dataset_name=dataset_name)
        # Populate the dataset with our examples.
        # `inputs` are extracted from the 'question' key, `outputs` from the 'response' key.
        client.create_examples(
            inputs=[{"question": ex["question"]} for ex in examples],
            outputs=[{"response": ex["response"]} for ex in examples],
            dataset_id=dataset.id,  # Associate examples with the created dataset.
        )

    experiment_results = await client.aevaluate(
        run_graph,  # The asynchronous function that runs our graph and returns its output
        data=dataset_name,  # The name of the LangSmith dataset to use for inputs and references
        evaluators=[
            final_answer_correct,
            correctness_evaluator,
        ],  # List of evaluator functions to apply
        experiment_prefix="Test",  # A prefix for the experiment name in LangSmith for better organization
        num_repetitions=1,  # Number of times to run each example (1 for quick testing)
        max_concurrency=5,  # Maximum concurrent runs to optimize evaluation speed
    )


asyncio.run(evaluate())
