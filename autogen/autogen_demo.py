"""
Quick AutoGen demo using a local Ollama model through an OpenAI-compatible endpoint.

What this is doing:
- We create two simple agents
- One agent acts like the researcher
- The other agent acts like the writer
- We let them take turns in a small team
- The goal is the same as the other framework demos:
  topic in -> key points -> polished final answer

This keeps the comparison fair across CrewAI, LangGraph, and AutoGen.
"""

import asyncio
import os

from dotenv import load_dotenv

# AutoGen AgentChat pieces
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

# AutoGen model client for OpenAI-compatible APIs
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load env vars from the shared root .env
# Since you're running from the project root, this is the simple version.
load_dotenv()


def build_model_client() -> OpenAIChatCompletionClient:
    """
    AutoGen uses a model client object instead of a direct LLM wrapper.

    Even though we're not using OpenAI itself, AutoGen's OpenAIChatCompletionClient
    can talk to OpenAI-compatible endpoints, which is how we connect to Ollama here.

    For AutoGen + Ollama, the important part is:
    - model should be the raw model name, like llama3.1:latest
    - base_url should point to the OpenAI-compatible endpoint, usually /v1
    """
    model = os.getenv("AUTOGEN_MODEL") 
    api_key = os.getenv("AUTOGEN_API_KEY") 
    base_url = os.getenv("AUTOGEN_BASE_URL") 

    print(f"\n[DEBUG] AutoGen model: {model}")
    print(f"[DEBUG] AutoGen base URL: {base_url}\n")

    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        
		model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown"
        }
    )


async def run_demo(topic: str) -> None:
    """
    This is the main AutoGen workflow.

    The pattern here is different from LangGraph:
    - In LangGraph, you explicitly define state and graph edges
    - In AutoGen, you define agents and let them talk in a team

    So the orchestration style here is conversational.
    """
    model_client = build_model_client()

    # First agent: generates the raw research points
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        description="Finds the most important ideas about a topic.",
        system_message=(
            "You are a research assistant.\n"
            "Your job is to analyze the user's topic and produce exactly 5 beginner-friendly key points.\n"
            "Rules:\n"
            "- Give exactly 5 points\n"
            "- Each point should be 1 to 2 sentences\n"
            "- Focus on concepts, not hype\n"
            "- Use numbered format 1 to 5\n"
            "- Do not add an introduction or conclusion"
        ),
    )

    # Second agent: turns the rough notes into a polished final response
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        description="Turns rough notes into a polished final answer.",
        system_message=(
            "You are a concise technical writer.\n"
            "You take the researcher's output and convert it into a clean final answer.\n"
            "Output format must be exactly:\n\n"
            "Topic: <topic>\n\n"
            "Key Points:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "4. ...\n"
            "5. ...\n\n"
            "Follow-up Questions:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n\n"
            "Rules:\n"
            "- Keep exactly 5 key points\n"
            "- Keep exactly 3 follow-up questions\n"
            "- Keep it clean and concise\n"
            "- Do not add extra sections"
        ),
    )

    # This stops the back-and-forth after a fixed number of messages.
    # For this tiny demo, we want a short, predictable run.
    termination = MaxMessageTermination(max_messages=4)

    # RoundRobinGroupChat means the agents take turns in order.
    # That's the simplest team pattern for a first demo.
    team = RoundRobinGroupChat(
        participants=[researcher, writer],
        termination_condition=termination,
    )

    task = (
        f"Topic: {topic}\n\n"
        "Step 1: researcher should produce exactly 5 beginner-friendly key points.\n"
        "Step 2: writer should turn that into the final formatted answer.\n"
        "Step 3: The writer must list follow-up questions only and must not answer them."

    )

    result = await team.run(task=task)

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
	
    # The final result contains the conversation messages.
    # Usually the last message is what we want to inspect for the final answer.
    if result.messages:
        print(result.messages[-1].content)
    else:
        print("No messages returned.")

    print("=" * 80)

    # Good cleanup habit for the model client
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run_demo("Agentic AI frameworks for beginner prototyping"))