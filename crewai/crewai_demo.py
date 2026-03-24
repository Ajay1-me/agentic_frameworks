"""
Test 2 workflows: The first workflow demonstrates basic agent-task orchestration, while the second shows multi-agent coordination with context passing and dependency chaining
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
    return LLM(
        model=os.getenv("MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.2, #0.2 keeps output more consistent and less random,
    )


# For the workflow 1, we keep it simple by having a Researcher that pulls out the most useful ideas
# and a Writer that formats and improves the final result
def build_agents_wf1(shared_llm: LLM):
    researcher = Agent(
        role="Research Analyst",
        goal="Identify the most important ideas about a topic and explain them clearly.",
        backstory=(
            "You are a focused research assistant. "
            "You read a topic carefully, identify the core concepts, "
            "and extract the most useful points for a beginner."
        ),
        llm=shared_llm,
        verbose=True #helps to see more of what is happening during execution.
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


def build_tasks_wf1(topic: str, researcher: Agent, writer: Agent):
    """
    Create the tasks for each agent.

    The first task generates structured research notes.
    The second task transforms those notes into the final output format.
    """
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Produce exactly 5 key points that help a beginner understand the topic.\n"
            "Each key point should be 1 to 2 sentences long.\n"
            "Focus on concepts, not hype.\n"
            "Avoid unnecessary jargon."
        ),
        # expected_output reinforces the details
        expected_output=(
            "A numbered list of exactly 5 key points about the topic, written clearly for a beginner."
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


def run_workflow_1(model: LLM, topic: str):
    """
    Main execution function with the following steps:
    1. Build the agents
    2. Build the tasks
    3. Create a crew
    4. Run the crew
    5. Print the final result
    """
    researcher, writer = build_agents_wf1(model)
    research_task, summary_task = build_tasks_wf1(topic, researcher, writer)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, summary_task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()


# Workflow 2 contains Advanced incident response workflow, Multiple specialized agents
# and demonstrates task dependency and more realistic orchestration
def build_agents_wf2(shared_llm: LLM):
    incident_analyst = Agent(
        role="Incident Analyst",
        goal="Analyze raw incident details and identify core problems.",
        backstory="You break down incidents into structured, actionable insights.",
        llm=shared_llm,
        verbose=True
    )

    impact_assessor = Agent(
        role="Impact Assessor",
        goal="Evaluate severity and impact across customers and operations.",
        backstory="You assess risk, scale, and urgency of incidents.",
        llm=shared_llm,
        verbose=True
    )

    planner = Agent(
        role="Restoration Planner",
        goal="Create a practical action plan to resolve the issue.",
        backstory="You convert analysis into clear operational steps.",
        llm=shared_llm,
        verbose=True
    )

    communicator = Agent(
        role="Communications Specialist",
        goal="Draft clear internal and customer updates.",
        backstory="You translate technical issues into simple communication.",
        llm=shared_llm,
        verbose=True
    )

    reviewer = Agent(
        role="QA Reviewer",
        goal="Ensure consistency and quality of final output.",
        backstory="You verify clarity, accuracy, and alignment.",
        llm=shared_llm,
        verbose=True
    )

    return incident_analyst, impact_assessor, planner, communicator, reviewer


def build_tasks_wf2(topic: str, a1, a2, a3, a4, a5):
    analyze = Task(
        description=f"Analyze this scenario:\n\n{topic}\n\nProvide summary, facts, risks.",
        expected_output="Structured incident analysis.",
        agent=a1
    )

    impact = Task(
        description="Assess severity and impact.",
        expected_output="Impact assessment with severity.",
        agent=a2,
        context=[analyze]
    )

    plan = Task(
        description="Create a restoration plan.",
        expected_output="Step-by-step recovery plan.",
        agent=a3,
        context=[analyze, impact]
    )

    comms = Task(
        description="Draft internal + customer communication.",
        expected_output="Two communication drafts.",
        agent=a4,
        context=[analyze, impact, plan]
    )

    review = Task(
        description="Review everything and produce final output.",
        expected_output="Final polished incident response.",
        agent=a5,
        context=[analyze, impact, plan, comms]
    )

    return [analyze, impact, plan, comms, review]


def run_workflow_2(model: LLM, topic: str):
    """Multi-agent dependent workflow."""
    agents = build_agents_wf2(model)
    tasks = build_tasks_wf2(topic, *agents)

    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()


def main():
    model = build_llm()

    topic_1 = "Agentic AI frameworks for beginner prototyping"
    topic_2 = "Power outage affecting 4,800 customers due to feeder issue near Substation 12."

    print("=" * 80)
    print("WORKFLOW 1: SIMPLE")
    print("=" * 80)
    output_1 = run_workflow_1(model, topic_1)
    print(output_1)

    print("\n" + "=" * 80)
    print("WORKFLOW 2: ADVANCED MULTI-AGENT")
    print("=" * 80)
    output_2 = run_workflow_2(model, topic_2)
    print(output_2)


if __name__ == "__main__":
    main()