"""
LangGraph demo using a local Ollama model.
This is just to understand how LangGraph works using a basic workflow
state -> nodes -> flow
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Dict, List
from langgraph.graph import StateGraph, START, END


# Load env variables
load_dotenv()


# Shared State
class GraphState(TypedDict):
    topic: str
    research_notes: str
    final_output: str

    # Workflow 2 fields
    ride_request: Dict
    options: List[Dict]
    selected_option: Dict
    is_valid: bool


# LLM
def build_llm() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.2,
    )


llm = build_llm()


# WORKFLOW 1
"""
def research_node(state: GraphState) -> dict:
    topic = state["topic"]

    prompt = f\"\"\"
You are a research assistant.

Topic: {topic}

- 5 key points
- 1–2 sentences each
\"\"\"

    response = llm.invoke(prompt)

    return {"research_notes": response.content}


def writer_node(state: GraphState) -> dict:
    topic = state["topic"]
    notes = state["research_notes"]

    prompt = f\"\"\"
Topic: {topic}
Notes: {notes}
Format clean output.
\"\"\"

    response = llm.invoke(prompt)

    return {"final_output": response.content}
"""


# WORKFLOW 2

# Node 1: Parse request
def parse_request_node(state: GraphState):
    return {
        "ride_request": state["ride_request"]
    }


# Node 2: Mock ride options (try using an API approach)
def fetch_options_node(state: GraphState):
    options = [
        {"service": "UberX", "price": 29, "eta": 6},
        {"service": "Uber Comfort", "price": 41, "eta": 4},
        {"service": "Lyft Standard", "price": 25, "eta": 14},
        {"service": "Lyft Priority", "price": 33, "eta": 7},
    ]

    return {"options": options}


# Node 3: Router
def route_node(state: GraphState):
    priority = state["ride_request"]["priority"]

    if priority == "cheapest":
        return "cheap_path"
    elif priority == "fastest":
        return "fast_path"
    else:
        return "balanced_path"


# Node 4A: Cheapest
def cheap_node(state: GraphState):
    best = min(state["options"], key=lambda x: x["price"])
    return {"selected_option": best}


# Node 4B: Fastest
def fast_node(state: GraphState):
    best = min(state["options"], key=lambda x: x["eta"])
    return {"selected_option": best}


# Node 4C: Balanced
def balanced_node(state: GraphState):
    best = min(state["options"], key=lambda x: x["price"] + x["eta"])
    return {"selected_option": best}


# Node 5: Validate
def validate_node(state: GraphState):
    option = state["selected_option"]
    req = state["ride_request"]

    valid = (
        option["price"] <= req["max_budget"]
        and option["eta"] <= req["max_wait_minutes"]
    )

    return {"is_valid": valid}


# Node 6: Fallback
def fallback_node(state: GraphState):
    # pick next best valid option
    req = state["ride_request"]

    valid_options = [
        o for o in state["options"]
        if o["price"] <= req["max_budget"]
        and o["eta"] <= req["max_wait_minutes"]
    ]

    if valid_options:
        best = min(valid_options, key=lambda x: x["price"])
        return {"selected_option": best}

    return {"selected_option": {"service": "None", "reason": "No valid rides"}}


# Node 7: Format output
def format_node(state: GraphState):
    option = state["selected_option"]

    if "reason" in option:
        output = "No suitable ride found within constraints."
    else:
        output = (
            f"Recommended Ride: {option['service']}\n"
            f"Price: ${option['price']}\n"
            f"ETA: {option['eta']} minutes"
        )

    return {"final_output": output}


# GRAPH BUILD
def build_graph_wf2():
    graph = StateGraph(GraphState)

    graph.add_node("parse", parse_request_node)
    graph.add_node("fetch", fetch_options_node)
    graph.add_node("cheap_path", cheap_node)
    graph.add_node("fast_path", fast_node)
    graph.add_node("balanced_path", balanced_node)
    graph.add_node("validate", validate_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("format", format_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "fetch")

    graph.add_conditional_edges(
        "fetch",
        route_node,
        {
            "cheap_path": "cheap_path",
            "fast_path": "fast_path",
            "balanced_path": "balanced_path",
        },
    )

    graph.add_edge("cheap_path", "validate")
    graph.add_edge("fast_path", "validate")
    graph.add_edge("balanced_path", "validate")

    graph.add_conditional_edges(
        "validate",
        lambda s: "valid" if s["is_valid"] else "invalid",
        {
            "valid": "format",
            "invalid": "fallback",
        },
    )

    graph.add_edge("fallback", "format")
    graph.add_edge("format", END)

    return graph.compile()


# RUN WORKFLOW 2
def run_workflow_2(model, ride_request):
    graph = build_graph_wf2()

    state: GraphState = {
        "topic": "",
        "research_notes": "",
        "final_output": "",
        "ride_request": ride_request,
        "options": [],
        "selected_option": {},
        "is_valid": False,
    }

    result = graph.invoke(state)
    return result["final_output"]


# MAIN
def main():
    model = build_llm()

    ride_request = {
        "pickup": "Sac State",
        "dropoff": "Airport",
        "priority": "cheapest",
        "max_budget": 30,
        "max_wait_minutes": 25,
    }

    print("=" * 80)
    print("WORKFLOW 2: UBER/LYFT ROUTER + VALIDATOR")
    print("=" * 80)
    output = run_workflow_2(model, ride_request)
    print(output)


if __name__ == "__main__":
    main()





"""



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

"""