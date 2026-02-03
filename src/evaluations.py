import uuid
from typing import Annotated

import rootpath
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.types import Command
from openevals.llm import (
    create_llm_as_judge,
)
from openevals.prompts import (
    CORRECTNESS_PROMPT,
)
from typing_extensions import TypedDict

from state_graph import get_state_graph

project_root = rootpath.detect()
load_dotenv(dotenv_path=project_root + "/.env", override=True)


async def run_graph(inputs: dict):
    multi_agent_final_graph = get_state_graph()
    graph = multi_agent_final_graph

    """Run graph and track the final response for evaluation."""
    thread_id = uuid.uuid4()
    configuration = {
        "configurable": {"thread_id": thread_id, "user_id": "10"},
        "recursion_limit": 50,
    }

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        config=configuration,
    )

    result = await graph.ainvoke(
        Command(resume="My customer ID is 10"),
        config={"configurable": {"thread_id": thread_id, "user_id": "10"}},
    )

    return {"response": result["messages"][-1].content}


llm = ChatOllama(
    model="qwen3",
    temperature=0,
)

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=llm,
)

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


async def final_answer_correct(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> bool:
    """Evaluate if the final response is equivalent to reference response."""
    # Construct the user prompt for the grader LLM, combining the question, ground truth, and student response.
    user = f"""QUESTION: {inputs["question"]}
    GROUND TRUTH RESPONSE: {reference_outputs["response"]}
    STUDENT RESPONSE: {outputs["response"]}"""

    grader_llm = llm.with_structured_output(Grade, method="json_schema", strict=True)

    grade = await grader_llm.ainvoke(
        [
            {"role": "system", "content": grader_instructions},
            {"role": "user", "content": user},
        ]
    )

    return grade["is_correct"]


#
# async def evaluate():
#     client = (
#         Client()
#     )  # Initialize the LangSmith client. This will connect to your LangSmith account.
#
#     examples = [
#         {
#             "question": "My name is Aaron Mitchell. My number associated with my account is +1 (204) 452-6452. I am trying to find the invoice number for my most recent song purchase. Could you help me with it?",
#             "response": "The Invoice ID of your most recent purchase was 342.",
#         },
#         # {
#         #     "question": "I'd like a refund.",
#         #     "response": "I need additional information to help you with the refund. Could you please provide your customer identifier so that we can fetch your purchase history?",
#         # },
#         # {
#         #     "question": "Who recorded Wish You Were Here again?",
#         #     "response": "Wish You Were Here is an album by Pink Floyd",
#         # },
#         # {
#         #     "question": "What albums do you have by Coldplay?",
#         #     "response": "There are no Coldplay albums available in our catalog at the moment.",
#         # },
#     ]
#
#     dataset_name = (
#         "LangGraph 101 Multi-Agent: Final Response"  # Define a name for our dataset.
#     )
#
#     # Check if the dataset already exists in LangSmith to avoid recreation.
#     if not client.has_dataset(dataset_name=dataset_name):
#         # If not, create the dataset.
#         dataset = client.create_dataset(dataset_name=dataset_name)
#         # Populate the dataset with our examples.
#         # `inputs` are extracted from the 'question' key, `outputs` from the 'response' key.
#         client.create_examples(
#             inputs=[{"question": ex["question"]} for ex in examples],
#             outputs=[{"response": ex["response"]} for ex in examples],
#             dataset_id=dataset.id,  # Associate examples with the created dataset.
#         )
#
#     experiment_results = await client.aevaluate(
#         run_graph,  # The asynchronous function that runs our graph and returns its output
#         data=dataset_name,  # The name of the LangSmith dataset to use for inputs and references
#         evaluators=[
#             final_answer_correct,
#             correctness_evaluator,
#         ],  # List of evaluator functions to apply
#         experiment_prefix="Test",  # A prefix for the experiment name in LangSmith for better organization
#         num_repetitions=1,  # Number of times to run each example (1 for quick testing)
#         max_concurrency=5,  # Maximum concurrent runs to optimize evaluation speed
#     )
#
#
# asyncio.run(evaluate())
