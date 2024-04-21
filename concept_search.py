"""Concept search system for LLMs."""

from copy import copy
from dataclasses import asdict, dataclass
import random
import re
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    Protocol,
    Self,
    Sequence,
    TypeVar,
)

from colorama import Fore
# from langchain_community.cache import SQLiteCache
# from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

S_contra = TypeVar("S_contra", contravariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S")
T = TypeVar("T")

DEFAULT_YAML = YAML()
DEFAULT_YAML.default_flow_style = False
DEFAULT_YAML.default_style = "|"  # type: ignore
DEFAULT_YAML.allow_unicode = True
# set_llm_cache(SQLiteCache(".llm_cache.db"))


def dedent_and_strip(text: str) -> str:
    """Dedent and strip text."""
    return dedent(text).strip()


def query(
    message: str,
    temperature: float = 0,
    color: str = Fore.MAGENTA,
    printout: bool = True,
) -> str:
    """Query an LLM chat model. `preamble` is printed before the result."""
    model = ChatGroq(temperature=temperature, model_name="llama3-70b-8192")
    if printout:
        print(f"\033[1;34m{message}\033[0m")
    result = str(model.invoke([HumanMessage(message)]).content)
    if printout:
        print(f"{color}{result}{Fore.RESET}")
    return result


@dataclass
class Definition:
    """Definition of some concept."""

    name: str
    definition: str

    def __eq__(self, other: Any) -> bool:
        """Check if two definitions are equal."""
        if not isinstance(other, Definition):
            return False
        return self.name == other.name


class Executor(Protocol, Generic[S_contra, T_co]):
    """Protocol for executing a prompt and returning the output."""

    def __call__(self, case: S_contra, concepts: Sequence[Definition]) -> T_co:
        raise NotImplementedError


class Outcome(Protocol):
    """Outcome for the result of an evaluation."""

    score: float
    max_score: float
    feedback: str


class Evaluator(Protocol, Generic[S_contra, T_contra]):
    """Evaluator protocol for checking the output of a prompt."""

    def __call__(self, case: S_contra, output: T_contra) -> Outcome:
        raise NotImplementedError


def format_as_yaml_str(
    data: Mapping[str, Any] | Sequence[Any], yaml: YAML = DEFAULT_YAML
) -> str:
    """Dump yaml as a string."""
    yaml.dump(data, stream := StringIO())
    return stream.getvalue().strip()


def format_definitions(definitions: Sequence[Definition]) -> str:
    """Format a list of definitions as a string."""
    return format_as_yaml_str([asdict(definition) for definition in definitions])

    # return "\n".join(
    #     f"- {definition.name}: {definition.definition}" for definition in definitions
    # )


def extract_block(text: str, block_type: str) -> str | None:
    """Extract a code block from the text."""
    pattern = (
        r"```{block_type}\n(.*?)```".format(  # pylint:disable=consider-using-f-string
            block_type=block_type
        )
    )
    match = re.search(pattern, text, re.DOTALL)
    return match[1].strip() if match else None


def query_and_extract(
    instructions: str, block_type: str, query_kwargs: Mapping[str, Any] | None = None
) -> str:
    """Query the model and extract a block from the output."""
    output = query(instructions, **(query_kwargs or {}))
    output = extract_block(output, block_type)
    assert output, f"Could not extract `{block_type}` block from:\n{output}"
    return output


def generate_concept_updates(
    defined_concepts: Sequence[Definition],
    score: float,
    max_score: float,
    feedback: str,
) -> str:
    """Generate concept updates based on the output of the evaluator."""
    defined_concepts_str = (
        format_definitions(defined_concepts) or "No concepts defined."
    )
    instructions = """
    # MISSION
    You are an advanced system that defines CONCEPTs to help simplify solving a TASK for a TASK_AGENT.

    ## DEFINITIONS
    - CONCEPT: Some definition that help perform a TASK. CONCEPTS are varied, ranging from general ideas to specific technical details. CONCEPTs can be standalone, but usually reference other CONCEPTs to for a rich web of understanding. CONCEPTS do *not* have to be human-understandable, so can be quite creative.
    - TASK_AGENT: An agent that can perform tasks, aided by CONCEPTs defined by you. The TASK_AGENT is capable but mechanical, and it needs precise definitions to operate effectively.
    - TASK: A task that the TASK_AGENT must solve, using the CONCEPTs you define. The TASK isn't necessarily a single task, but a class of related tasks.

    ## DEFINED_CONCEPTS
    Here are the CONCEPTs that have been defined so far for the TASK_AGENT:
    ```start_of_defined_concepts
    {defined_concepts}
    ```end_of_defined_concepts

    ## TASK_BACKGROUND
    You have no general information about the TASK—you will need to figure out the parameters of the TASK based on the information provided in the TASK_AGENT_EVALUATION section.

    ## TASK_AGENT_EVALUATION
    Based on the DEFINED_CONCEPTS, the TASK_AGENT attempted to perform the TASK, with the following feedback:
    ```start_of_task_agent_evaluation
    score: {score}/{max_score}
    feedback: |-
    {feedback}
    ```end_of_task_agent_evaluation
    
    ## INSTRUCTIONS
    There are 3 main phases to your task:

    ### PHASE 1: REASONING_GENERATION
    Generate a nested reasoning structure in YAML format for yourself to sequentially process the information above to figure out how to update the DEFINED_CONCEPTS to improve the TASK_AGENT's performance. Output this in the following format:
    ```start_of_reasoning_yaml
    {{reasoning_structure}}
    ```end_of_reasoning_yaml

    ### PHASE 2: REASONING_PROCESSING
    Follow the reasoning structure you generated above, posting output for each part of the reasoning in the following YAML format:
    ```start_of_reasoning_output_yaml
    {{reasoning_output}}
    ```end_of_reasoning_output_yaml

    ### PHASE 3: CONCEPT_UPDATE
    Create updates for DEFINED_CONCEPTS based on the output from REASONING_PROCESSING. The updates should be in the following format in YAML:
    ```start_of_concept_updates_yaml
    - concept_name: "{{CONCEPT_NAME_1}}"
      action: "{{ACTION_1}}"
      concept_definition: |-
        {{definition_1}}
    - concept_name: "{{CONCEPT_NAME_2}}"
      action: "{{ACTION_2}}"
      concept_definition: |-
        {{definition_2}}
    # [...etc.]
    ```end_of_concept_updates_yaml
    Where:
    `concept_name` must be all capitalized and in snake case.
    `action` must be one of the following: "ADD", "REMOVE", or "MODIFY".
    `concept_definition` is either the new/modifed definition, or an empty string if the concept is to be removed. 

    Post the output to all 3 phases in the same response.
    """
    instructions = dedent_and_strip(instructions).format(
        defined_concepts=defined_concepts_str,
        score=score,
        max_score=max_score,
        feedback=feedback,
    )
    return query_and_extract(
        instructions,
        block_type="start_of_concept_updates_yaml",
        query_kwargs={"temperature": 1},
    )


@dataclass
class ConceptUpdate:
    """Update to a concept."""

    concept: Definition
    action: Literal["ADD", "REMOVE", "MODIFY"]

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Self:
        """Create from a dictionary."""
        action = data["action"]
        assert action in ["ADD", "REMOVE", "MODIFY"], f"Invalid action: {action}"
        return cls(
            concept=Definition(data["concept_name"], data["concept_definition"]),
            action=action,  # type: ignore
        )


def update_concepts(
    defined_concepts: Sequence[Definition], concept_updates: Sequence[ConceptUpdate]
) -> list[Definition]:
    """Apply updates to the defined concepts."""
    updated_concepts = list(copy(defined_concepts))
    for update in concept_updates:
        if update.action in ["REMOVE"]:
            updated_concepts.remove(update.concept)
        if update.action in ["ADD"]:
            updated_concepts.append(update.concept)
        if update.action in ["MODIFY"]:
            for idx, concept in enumerate(updated_concepts):
                if concept == update.concept:
                    updated_concepts[idx] = update.concept
    return updated_concepts


def optimize_concepts(
    test_cases: Sequence[S],
    execute: Executor[S, T],
    evaluate: Evaluator[S, T],
    condense_feedback: Callable[[Sequence[Outcome]], str],
    success_threshold: float,
    max_rounds: int,
    initial_concepts: Sequence[Definition] | None = None,
) -> tuple[list[Definition] | None, float, float]:
    """Optimize a prompt by evaluating it against a set of test cases and performing concept search."""
    assert max_rounds > 0, "max_rounds must be greater than 0"
    assert 0 <= success_threshold <= 1, "success_threshold must be between 0 and 1"

    def get_results(
        defined_concepts: Sequence[Definition],
    ) -> tuple[float, float, str]:
        outcomes: list[Outcome] = []
        for test_case in test_cases:
            output = execute(test_case, defined_concepts)
            outcome = evaluate(test_case, output)
            outcomes.append(outcome)
        score = sum(outcome.score for outcome in outcomes)
        max_score = sum(outcome.max_score for outcome in outcomes)
        feedback = condense_feedback(outcomes)
        return score, max_score, feedback

    concepts = list(copy(initial_concepts or []))
    score, max_score, feedback = get_results(concepts)
    for _ in range(max_rounds):
        concept_updates = generate_concept_updates(concepts, score, max_score, feedback)
        concept_updates = [
            ConceptUpdate.from_dict(update)  # type: ignore
            for update in DEFAULT_YAML.load(concept_updates)  # type: ignore
        ]
        updated_concepts = update_concepts(concepts, concept_updates)
        updated_score, updated_max_score, updated_feedback = get_results(
            updated_concepts
        )
        if updated_score / updated_max_score > score / max_score:
            concepts, score, max_score, feedback = (
                updated_concepts,
                updated_score,
                updated_max_score,
                updated_feedback,
            )
            if score / max_score >= success_threshold:
                break
    concepts = None if concepts == initial_concepts else concepts
    return concepts, score, max_score


@dataclass
class ABOutcome:
    """Outcome for the A# B# problem."""

    score: float
    max_score: float
    feedback: str


def evaluate_ab(case: Sequence[str], output: str) -> ABOutcome:
    """Evaluate the output of a case for the A# B# problem."""
    answer = copy(list(case))
    mappings = {
        ("A#", "#A"): (),
        ("B#", "#B"): (),
        ("B#", "#A"): ("#A", "B#"),
        ("A#", "#B"): ("#B", "A#"),
    }
    complete = False
    while not complete:
        for idx in range(len(answer) - 1):
            token_pair = answer[idx], answer[idx + 1]
            if token_pair in mappings:
                replacement = mappings[token_pair]
                answer = answer[:idx] + list(replacement) + answer[idx + 2 :]
                break
        else:
            complete = True
    expected_output = " ".join(answer)
    score = float(output == expected_output)
    feedback = """
    - input: {input}
      score: {score}/1
      expected_output: {expected_output}
      actual_output: {output}
    """
    feedback = dedent_and_strip(feedback).format(
        input=" ".join(case),
        score=score,
        expected_output=expected_output,
        output=output,
    )
    return ABOutcome(score=score, max_score=1, feedback=feedback)


def execute_ab(case: Sequence[str], concepts: Sequence[Definition]) -> str:
    """Execute a case for the A# B# problem."""
    case_text = " ".join(case)
    instructions = """
    # MISSION
    You are executing a task based on CONCEPTs defined below.

    ## CONCEPTS
    Here are CONCEPTs related to the task:
    ```start_of_concepts
    {defined_concepts}
    ```end_of_concepts

    ## TASK
    Resolve the following input sequence:
    ```start_of_input
    {input}
    ```end_of_input

    ## INSTRUCTIONS
    There are 3 main phases to performing the TASK:

    ### PHASE 1: REASONING_GENERATION
    Generate a nested reasoning structure in YAML format for yourself to sequentially process the information above, so that you can perform the TASK, using the CONCEPTs defined above. Output this in the following format:
    ```start_of_reasoning_yaml
    {{reasoning_structure}}
    ```end_of_reasoning_yaml

    ### PHASE 2: REASONING_PROCESSING
    Follow the reasoning structure above, posting output for each part of the reasoning in the following format:
    ```start_of_reasoning_output_yaml
    {{reasoning_output}}
    ```end_of_reasoning_output_yaml

    ### PHASE 3: FINAL_OUTPUT
    Output the final output sequence in the following YAML format:
    ```start_of_final_output
    {{output_sequence}}
    ```end_of_final_output
    Post the output of all 3 phases in the same response.
    """
    instructions = dedent_and_strip(instructions).format(
        defined_concepts=format_definitions(concepts),
        input=case_text,
    )
    return query_and_extract(instructions, block_type="start_of_final_output")


def condense_ab_feedback(outcomes: Sequence[Outcome]) -> str:
    """Condense feedback for the A# B# problem."""
    outcomes = sorted(outcomes, key=lambda outcome: outcome.score, reverse=True)
    return "\n".join(outcome.feedback for outcome in outcomes)


RESOLUTION_EXAMPLE = """
Starting sequence: B# A# #B #A B#
1. Apply RULE_4: B# #B A# #A B#
2. Apply RULE_2: A# #A B#
3. Apply RULE_1: B#
""".strip()
INITIAL_AB_CONCEPTS = [
    Definition("TOKEN", "There are 4 TOKENs in this problem: A#, B#, #A, and #B"),
    Definition(
        "SEQUENCE",
        "A SEQUENCE is a list of TOKENs. Example of a SEQUENCE: A# B# #B A#",
    ),
    Definition(
        "SEQUENCE_RESOLUTION",
        "SEQUENCE_RESOLUTION is the process of resolving an initial SEQUENCE into a final output SEQUENCE, based certain rules.",
    ),
    Definition(
        "EMPTY_SEQUENCE",
        "An EMPTY_SEQUENCE is a SEQUENCE that contains no TOKENs. It is denoted as (empty) when standalone; as part of a SEQUENCE, it's not displayed.",
    ),
    Definition(
        "RESOLUTION_RULE",
        "A particular rule that is applied during SEQUENCE_RESOLUTION, converting certain sub-SEQUENCEs into others. A RESOLUTION_RULE has the following format: <SEQUENCE_1> -> <SEQUENCE_2>, which means that whenever you see <SEQUENCE_1>, you replace it with <SEQUENCE_2> during SEQUENCE RESOLUTION.",
    ),
    Definition("RULE_1", "A# #A -> (empty)"),
    Definition("RULE_2", "B# #B -> (empty)"),
    Definition("RULE_3", "B# #A -> #A B#"),
    Definition("RULE_4", "A# #B -> #B A#"),
    Definition("RESOLUTION_EXAMPLE", RESOLUTION_EXAMPLE),
]


def generate_distinct_lists(
    values: Sequence[Any], num_lists: int, list_length: int, seed: int = 0
):
    """Generates distinct random lists of some length using the given set of values."""
    random.seed(seed)
    max_possible_combinations = len(values) ** list_length
    if num_lists > max_possible_combinations:
        raise ValueError(
            "Not enough distinct lists can be generated with the given parameters."
        )
    generated_lists: set[tuple[Any, ...]] = set()
    while len(generated_lists) < num_lists:
        # Generate a random list of length n
        new_list = tuple(random.choices(values, k=list_length))
        generated_lists.add(new_list)  # Set ensures all entries are unique
        # Handle the case where space is saturated and looping infinitely
        if len(generated_lists) == max_possible_combinations:
            break
    return list(generated_lists)


# TEST_CASES_AB = [
#     ["A#", "B#", "#B"],
# ]
TEST_CASES_AB = generate_distinct_lists(
    ["A#", "B#", "#A", "#B"], num_lists=10, list_length=3
)


def main() -> None:
    """Run the main function."""
    optimized_concepts, score, max_score = optimize_concepts(
        test_cases=TEST_CASES_AB,
        execute=execute_ab,
        evaluate=evaluate_ab,
        condense_feedback=condense_ab_feedback,
        success_threshold=0.9,
        max_rounds=3,
        initial_concepts=INITIAL_AB_CONCEPTS,
    )
    print(
        format_definitions(optimized_concepts) if optimized_concepts else "No updates."
    )
    print(f"Score: {score}/{max_score}")


if __name__ == "__main__":
    main()
