"""
Test 2 workflows:

1. Simple workflow
   - topic -> 5 key points -> 3 follow-up questions

2. Slightly more complex workflow
   - classify the topic
   - use a tool to choose the response style
   - generate the response
   - validate the response format
   - retry once if the format is wrong
"""

import os
import re
from typing import Optional
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


# Load the shared root .env
load_dotenv()


def build_model() -> ChatOllama:
    model = os.getenv("OLLAMA_MODEL") 
    base_url = os.getenv("BASE_URL") 

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.2,
    )



# Tools for workflow 2

# Classify a topic as technical, business, or mixed.
@tool
def classify_topic(topic: str) -> str:
    topic_lower = topic.lower()

    technical_words = ["api", "architecture", "system", "search", "database", "vector", "llm", "agent"]
    business_words = ["budget", "strategy", "stakeholder", "roi", "adoption", "value", "operations"]

    technical_hits = sum(1 for word in technical_words if word in topic_lower)
    business_hits = sum(1 for word in business_words if word in topic_lower)

    if technical_hits > business_hits:
        return "technical"
    if business_hits > technical_hits:
        return "business"
    return "mixed"



#Return writing instructions for a given category.
@tool
def get_style_guide(category: str) -> str:
    guides = {
        "technical": (
            "Write with an implementation mindset. Mention architecture, components, "
            "tradeoffs, constraints, and practical next steps."
        ),
        "business": (
            "Write with a business mindset. Mention value, adoption, risk, stakeholders, "
            "and operational impact."
        ),
        "mixed": (
            "Balance technical and business perspectives. Keep it practical and understandable "
            "for both technical and non-technical readers."
        ),
    }
    return guides.get(category, guides["mixed"])


@tool
def validate_output_format(output: str) -> str:
    """Check whether the output has Topic, exactly 5 key points, and exactly 3 follow-up questions."""
    has_topic = "Topic:" in output
    has_key_points_header = "Key Points:" in output
    has_questions_header = "Follow-up Questions:" in output

    key_points_count = 0
    question_count = 0

    if has_key_points_header:
        key_section = output.split("Key Points:", 1)[-1]
        if "Follow-up Questions:" in key_section:
            key_section = key_section.split("Follow-up Questions:", 1)[0]
        key_points_count = len(re.findall(r"^\s*\d+\.", key_section, flags=re.MULTILINE))

    if has_questions_header:
        question_section = output.split("Follow-up Questions:", 1)[-1]
        question_count = len(re.findall(r"^\s*\d+\.", question_section, flags=re.MULTILINE))

    is_valid = (
        has_topic
        and has_key_points_header
        and has_questions_header
        and key_points_count == 5
        and question_count == 3
    )

    if is_valid:
        return "VALID"

    return (
        "INVALID\n"
        f"has_topic={has_topic}\n"
        f"has_key_points_header={has_key_points_header}\n"
        f"has_questions_header={has_questions_header}\n"
        f"key_points_count={key_points_count}\n"
        f"question_count={question_count}"
    )


def extract_final_text(result: dict) -> str:
    """
    Pull the final assistant text out of the agent response.
    """
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return "No final text returned."


def run_workflow_1(model: ChatOllama, topic: str) -> str:
    
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a concise research assistant.\n"
            "Given a topic, produce this exact structure:\n\n"
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
            "- exactly 5 key points\n"
            "- exactly 3 follow-up questions\n"
            "- do not answer the questions\n"
            "- no extra sections\n"
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Topic: {topic}",
                }
            ]
        }
    )
    return extract_final_text(result)


def run_workflow_2(model: ChatOllama, topic: str) -> str:
  
    agent = create_agent(
        model=model,
        tools=[classify_topic, get_style_guide, validate_output_format],
        system_prompt=(
            "You are a project brief assistant.\n"
            "Your job is to create structured output for a topic.\n\n"
            "You should:\n"
            "1. Classify the topic using the classify_topic tool\n"
            "2. Use get_style_guide to choose the right framing\n"
            "3. Generate the response in the required structure\n"
            "4. Use validate_output_format on your own draft\n"
            "5. If validation says INVALID, fix the draft and validate once more\n\n"
            "Required structure:\n"
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
            "- exactly 5 key points\n"
            "- exactly 3 follow-up questions\n"
            "- do not answer the questions\n"
            "- no extra sections\n"
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Create a project brief for this topic: {topic}\n"
                        "Make it practical and audience-aware."
                    ),
                }
            ]
        }
    )
    return extract_final_text(result)


def main():
    model = build_model()

    topic_1 = "Agentic AI frameworks for beginner prototyping"
    topic_2 = "Build an internal HR document search tool for employees"

    print("=" * 80)
    print("WORKFLOW 1: SIMPLE")
    print("=" * 80)
    output_1 = run_workflow_1(model, topic_1)
    print(output_1)

    print("\n" + "=" * 80)
    print("WORKFLOW 2: ROUTER + VALIDATOR")
    print("=" * 80)
    output_2 = run_workflow_2(model, topic_2)
    print(output_2)


if __name__ == "__main__":
    main()