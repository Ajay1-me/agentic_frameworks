"""
LangGraph demo using a local Ollama model.

What we're trying to do here:
- Take a topic
- Generate 5 key points about it
- Turn that into a clean final response with follow-up questions

This is just to understand how LangGraph works:
state -> nodes -> flow
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# Load env variables
load_dotenv()


# ---------------------------
# Shared state
# ---------------------------
# Think of this like a shared "data object" that moves through the graph.
# Every node can read from it and update parts of it.

class GraphState(TypedDict):
    topic: str
    research_notes: str
    final_output: str


# LLM setup: We're plugging in Ollama via ChatOllama.

def build_llm() -> ChatOllama:
    model = os.getenv("OLLAMA_MODEL")
    base_url = os.getenv("BASE_URL") 

    #print(f"\n[DEBUG] Using model: {model}")
    #print(f"[DEBUG] Using base URL: {base_url}\n")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.2,
    )


#create one shared LLM instance (so both nodes reuse it)
llm = build_llm()



# Node 1: Research
# This is the first step in our flow.
# It reads the topic and generates raw notes.

def research_node(state: GraphState) -> dict:
    topic = state["topic"]

    prompt = f"""
You are a research assistant.

Topic: {topic}

Instructions:
- Give exactly 5 key points
- Keep them beginner-friendly
- 1–2 sentences each
- No intro, no conclusion
- Just a numbered list (1–5)
""".strip()

    response = llm.invoke(prompt)

    # We only update part of the state here
    return {
        "research_notes": response.content
    }


# Node 2: Writer
# This takes the rough notes and cleans them up into final output.

def writer_node(state: GraphState) -> dict:
    topic = state["topic"]
    notes = state["research_notes"]

    prompt = f"""
You are a concise technical writer.

Topic: {topic}

Notes:
{notes}

Format the output exactly like this:

Topic: {topic}

Key Points:
1. ...
2. ...
3. ...
4. ...
5. ...

Follow-up Questions:
1. ...
2. ...
3. ...

Rules:
- Keep it clean and short
- Exactly 5 points
- Exactly 3 questions
- No extra text
""".strip()

    response = llm.invoke(prompt)

    return {
        "final_output": response.content
    }


# Build the graph
# Notice this is where LangGraph becomes different from CrewAI.

# were explicitly saying to Run this node, then this node, then stop

def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)

    # Define flow
    graph.add_edge(START, "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def run_demo(topic: str):
    graph = build_graph()

    # Initial state: only topic is filled
    state: GraphState = {
        "topic": topic,
        "research_notes": "",
        "final_output": "",
    }

    result = graph.invoke(state)

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(result["final_output"])
    print("=" * 80)


if __name__ == "__main__":
    run_demo("Agentic AI frameworks for beginner prototyping")