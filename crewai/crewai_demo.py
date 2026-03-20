"""
A minimal CrewAI demo for quickly learning the framework.

What this script does:
1. Loads environment variables from a .env file
2. Creates one shared LLM configuration
3. Creates two agents:
   - Researcher
   - Writer
4. Creates two tasks:
   - Research task
   - Writing task
5. Runs both tasks sequentially as one crew
6. Prints the final output to the terminal

This is intentionally small so you can understand CrewAI's core abstractions fast.
"""

import os
from dotenv import load_dotenv

# Core CrewAI classes:
# - Agent: defines a role/persona that performs work
# - Task: defines a specific piece of work assigned to an agent
# - Crew: orchestrates the execution of multiple tasks/agents
# - Process: controls how tasks are run (sequential is easiest to start with)

from crewai import Agent, Task, Crew, Process, LLM


# Load environment variables from .env into the current process
load_dotenv()

# LLM setup
def build_llm() -> LLM:
    
    model = os.getenv("MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2 #0.2 keeps output more consistent and less random,
    )

    return llm


def build_agents(shared_llm: LLM):
    """
    Create the agents.

    We keep them very simple:
    - Researcher: pulls out the most useful ideas
    - Writer: formats and improves the final result

    verbose=True helps you see more of what is happening during execution.
    That is useful while learning.
    """
    researcher = Agent(
        role="Research Analyst",
        goal="Identify the most important ideas about a topic and explain them clearly.",
        backstory=(
            "You are a focused research assistant. "
            "You read a topic carefully, identify the core concepts, "
            "and extract the most useful points for a beginner."
        ),
        llm=shared_llm,
        verbose=True
    )

    writer = Agent(
        role="Technical Writer",
        goal="Turn raw research notes into a clean, polished summary.",
        backstory=(
            "You are a concise technical writer. "
            "You take rough notes and convert them into clear, well-structured output."
        ),
        llm=shared_llm,
        verbose=True
    )

    return researcher, writer


def build_tasks(topic: str, researcher: Agent, writer: Agent):
    """
    Create the tasks for each agent.

    The first task generates structured research notes.
    The second task transforms those notes into the final output format.

    expected_output is optional but useful because it nudges the model
    into producing a cleaner, more predictable result.
    """

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Produce exactly 5 key points that help a beginner understand the topic.\n"
            "Each key point should be 1 to 2 sentences long.\n"
            "Focus on concepts, not hype.\n"
            "Avoid unnecessary jargon."
        ),
        expected_output=(
            "A numbered list of exactly 5 key points about the topic, "
            "written clearly for a beginner."
        ),
        agent=researcher
    )

    summary_task = Task(
        description=(
            "Take the research notes from the previous task and turn them into a polished final response.\n\n"
            "Output format:\n"
            "1. A title line: 'Topic: <topic>'\n"
            "2. A section called 'Key Points:' with exactly 5 numbered bullets\n"
            "3. A section called 'Follow-up Questions:' with exactly 3 numbered questions\n\n"
            "Keep the writing concise, clear, and professional."
        ),
        expected_output=(
            "A polished response containing a topic title, 5 key points, "
            "and 3 follow-up questions."
        ),
        agent=writer
    )

    return research_task, summary_task


def run_demo(topic: str):
    """
    Main execution function.

    Steps:
    1. Build the shared LLM
    2. Build the agents
    3. Build the tasks
    4. Create a crew
    5. Run the crew
    6. Print the final result
    """
    shared_llm = build_llm()
    researcher, writer = build_agents(shared_llm)
    research_task, summary_task = build_tasks(topic, researcher, writer)

    # The crew is the orchestrator that runs tasks and coordinates agents.
    # Process.sequential means:
    # - first run research_task
    # - then run summary_task
    # This is the easiest process to understand when starting out.
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, summary_task],
        process=Process.sequential,
        verbose=True
    )

    # kickoff() starts the execution.
    # We do not need complex inputs yet because we already embedded topic
    # directly into the task descriptions for simplicity.
    result = crew.kickoff()

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(result)
    print("=" * 80)


if __name__ == "__main__":
    # test a different input here
    topic = "Agentic AI frameworks for beginner prototyping"

    run_demo(topic)