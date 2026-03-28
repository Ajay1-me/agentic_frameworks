"""
Quick AutoGen demo using a local Ollama model through an OpenAI-compatible endpoint.
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


# Load env vars
load_dotenv()


def build_model_client() -> OpenAIChatCompletionClient:
    model = os.getenv("AUTOGEN_MODEL") 
    api_key = os.getenv("AUTOGEN_API_KEY") 
    base_url = os.getenv("AUTOGEN_BASE_URL") 

    #print(f"\n[DEBUG] AutoGen model: {model}")
    #print(f"[DEBUG] AutoGen base URL: {base_url}\n")

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


# WORKFLOW 1
"""
async def run_demo(topic: str) -> None:
    
    The pattern here is different from LangGraph:
    - In LangGraph, I had to explicitly define state and graph edges
    - but in AutoGen, define agents and let them talk in a team

    So the orchestration style here is conversational.
    
    model_client = build_model_client()

    # First agent: generates the raw research points
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        description="Find the most important ideas about a topic.",
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

    # This stops the back and forth after a fixed number of messages.
    # For this tiny demo we want a short and predictable run.
    termination = MaxMessageTermination(max_messages=4)

    # RoundRobinGroupChat means the agents take turns in order.
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
"""


# WORKFLOW 2
async def run_workflow_2(model_client: OpenAIChatCompletionClient, application_request: dict) -> str:
    """
    This is a Multi-agent job application workflow.

    The team will:
    1. Analyze the job posting
    2. Decide what parts of the candidate profile matter most
    3. Tailor resume bullets
    4. Draft a short targeted cover letter
    5. Review and assemble the final application packet
    """

    job_analyst = AssistantAgent(
        name="job_analyst",
        model_client=model_client,
        description="Extracts the most important requirements and signals from the job posting.",
        system_message=(
            "You are a job analyst. "
            "Read the job posting carefully and identify the most important qualifications, skills, "
            "responsibilities, and themes. "
            "Distinguish between must-have requirements and nice-to-have preferences. "
            "Keep your analysis structured and practical for downstream application tailoring."
        ),
    )

    resume_strategist = AssistantAgent(
        name="resume_strategist",
        model_client=model_client,
        description="Chooses which parts of the candidate profile should be emphasized.",
        system_message=(
            "You are a resume strategist. "
            "Use the job analysis and candidate background to decide what should be emphasized. "
            "Prioritize relevance, business impact, technical alignment, and leadership signals. "
            "Select the strongest experiences and explain why they should be surfaced."
        ),
    )

    resume_tailor = AssistantAgent(
        name="resume_tailor",
        model_client=model_client,
        description="Rewrites candidate resume bullets to align better with the role.",
        system_message=(
            "You are a resume tailoring specialist. "
            "Rewrite resume bullets so they align with the role while staying truthful to the candidate's background. "
            "Do not invent technologies, job titles, or accomplishments. "
            "Improve clarity, relevance, and impact. "
            "Favor concise bullets with action + impact. "
            "Output exactly 5 tailored resume bullets based on the candidate background and job analysis."
        ),
    )

    cover_letter_writer = AssistantAgent(
        name="cover_letter_writer",
        model_client=model_client,
        description="Drafts a concise cover letter tailored to the role.",
        system_message=(
            "You are a professional cover letter writer. "
            "Write a concise, targeted cover letter that sounds polished and credible. "
            "Connect the candidate's real background to the role's needs. "
            "Keep it professional, focused, and specific. "
            "Do not be overly dramatic or generic."
        ),
    )

    application_reviewer = AssistantAgent(
        name="application_reviewer",
        model_client=model_client,
        description="Reviews alignment and assembles the final application packet.",
        system_message=(
            "You are an application reviewer and final editor. "
            "Review the team's work for consistency, truthfulness, alignment to the job, and professionalism. "
            "Then produce the final application packet in exactly this format:\\n\\n"
            "Job Fit Summary:\\n"
            "<4-6 sentences>\\n\\n"
            "Top Matching Qualifications:\\n"
            "1. ...\\n"
            "2. ...\\n"
            "3. ...\\n"
            "4. ...\\n"
            "5. ...\\n\\n"
            "Tailored Resume Bullets:\\n"
            "1. ...\\n"
            "2. ...\\n"
            "3. ...\\n"
            "4. ...\\n"
            "5. ...\\n\\n"
            "Cover Letter:\\n"
            "<short cover letter>\\n\\n"
            "Application Risks or Gaps:\\n"
            "1. ...\\n"
            "2. ...\\n"
            "3. ...\\n\\n"
            "Rules:\\n"
            "- Keep the tailored resume bullets concise\\n"
            "- Keep the cover letter to around 180 to 250 words\\n"
            "- Do not invent qualifications\\n"
            "- Do not add extra sections\\n"
            "- Do not include: greetings (eg: Hello, Hi)\\n"
            "- closing statements (e.g., Best regards, Thank you)\\n"
            "- concluding paragraphs outside the required sections\\n"
            "- any text before or after the defined sections"
        ),
    )

    termination = MaxMessageTermination(max_messages=10)

    team = RoundRobinGroupChat(
        participants=[
            job_analyst,
            resume_strategist,
            resume_tailor,
            cover_letter_writer,
            application_reviewer,
        ],
        termination_condition=termination,
    )

    task = f"""
You are collaborating on a job application packet.

JOB POSTING:
{application_request["job_posting"]}

CANDIDATE BACKGROUND:
Name: {application_request["candidate_name"]}

Professional Summary:
{application_request["candidate_summary"]}

Experience Highlights:
{application_request["experience_highlights"]}

Projects:
{application_request["projects"]}

Skills:
{application_request["skills"]}

TEAM OBJECTIVE:
Build a strong, truthful, role-aligned application packet.

TEAM STEPS:
1. job_analyst should identify the most important job requirements and priorities.
2. resume_strategist should decide which parts of the candidate background best match the role.
3. resume_tailor should produce exactly 5 tailored resume bullets based only on the provided background.
4. cover_letter_writer should draft a concise cover letter tailored to the role and candidate profile.
5. application_reviewer should review everything and produce the final application packet in the required format.

IMPORTANT RULES:
- Do not invent experience, credentials, or technologies not present in the candidate background.
- Prefer specificity over generic claims.
- Surface leadership, impact, and technical relevance where justified.
- The final answer should be polished enough to show thoughtful application preparation.
""".strip()

    result = await team.run(task=task)

    if result.messages:
        return result.messages[-1].content
    return "No messages returned."



# MAIN
async def main():
    model_client = build_model_client()

    application_request = {
        "candidate_name": "Ajaydeep Singh",
        "candidate_summary": (
            "Computer Science student with internship experience in enterprise technology, "
            "AI automation, semantic search, frontend development, and workflow optimization. "
            "Has led student tech organizations, built AI-focused demos, and worked on tools "
            "involving document intelligence, search, automation, and full-stack development."
        ),
        "experience_highlights": (
            "- Enterprise Technology Developer Intern with experience in AI automation, semantic search, and enterprise tooling\n"
            "- Front-End UI/UX Development Intern with experience improving interfaces and user workflows\n"
            "- Automation and Test Engineering internship experience involving process efficiency and testing support\n"
            "- Leadership experience as ACM President organizing technical events, hackathons, and company-facing activities"
        ),
        "projects": (
            "- Built a semantic search proof of concept over enterprise documents\n"
            "- Developed AI and automation workflows using modern frameworks\n"
            "- Worked on full-stack and frontend projects involving React, Python, and enterprise integrations\n"
            "- Participated in machine learning and applied AI projects"
        ),
        "skills": (
            "Python, Java, JavaScript, TypeScript, React, SQL, FastAPI, AI/ML concepts, "
            "semantic search, enterprise automation, UI/UX, teamwork, leadership, communication"
        ),
        "job_posting": (
            "Associate Software Engineer\n\n"
            "We are looking for an early-career software engineer to help build internal tools and digital products. "
            "The ideal candidate has experience with Python or JavaScript, understands APIs and data flows, "
            "can work across frontend and backend tasks, and is comfortable collaborating in a fast-moving team. "
            "Exposure to AI-powered applications, workflow automation, or enterprise systems is a plus. "
            "Strong communication, adaptability, and ownership mindset are highly valued."
        ),
    }

    print("=" * 80)
    print("WORKFLOW 2: JOB APPLICATION AGENT SYSTEM")
    print("=" * 80)
    output_2 = await run_workflow_2(model_client, application_request)
    print(output_2)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())



