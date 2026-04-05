from __future__ import annotations

import argparse
import os
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv
import os



load_dotenv()


def build_llm() -> LLM:
    os.environ["OPENAI_API_KEY"] = "ollama"

    return LLM(
    model="ollama/llama3.1",  
    base_url="http://localhost:11434"
)


def build_agents(llm: LLM) -> dict[str, Agent]:
    return {
        "jd_parser": Agent(
            role="JD Parser",
            goal="Extract the key role requirements, skills, experience, and responsibilities from the job description.",
            backstory="You turn raw job descriptions into concise hiring requirements that other agents can use.",
            llm=llm,
            verbose=True,
        ),
        "resume_analyzer": Agent(
            role="Resume Analyzer",
            goal="Summarize the candidate profile, skills, experience, strengths, and weak spots from the resume.",
            backstory="You read resumes carefully and convert them into structured hiring insights.",
            llm=llm,
            verbose=True,
        ),
        "ats_gap_finder": Agent(
            role="ATS Gap Finder",
            goal="Compare the resume against the job description and identify missing keywords, skills, and experience gaps.",
            backstory="You think like an applicant tracking system and a recruiter at the same time.",
            llm=llm,
            verbose=True,
        ),
        "resume_rewriter": Agent(
            role="Resume Rewriter",
            goal="Rewrite resume sections so they align better with the job description without inventing facts.",
            backstory="You improve resume wording, clarity, and keyword alignment while staying truthful.",
            llm=llm,
            verbose=True,
        ),
        "interview_question_generator": Agent(
            role="Interview Question Generator",
            goal="Create practical interview questions based on the candidate profile and the target role.",
            backstory="You produce targeted interview questions that test both fit and any suspected gaps.",
            llm=llm,
            verbose=True,
        ),
    }


def build_tasks(agents: dict[str, Agent]) -> list[Task]:
    jd_parser_task = Task(
        description=(
            "Read the job description below and extract the most important details.\n\n"
            "Job Description:\n{job_description}\n\n"
            "Return these sections:\n"
            "1. Target job title\n"
            "2. Must-have skills\n"
            "3. Nice-to-have skills\n"
            "4. Required experience\n"
            "5. Key responsibilities\n"
            "6. Top ATS keywords"
        ),
        expected_output="A clean structured summary of the job description with ATS-focused keywords.",
        agent=agents["jd_parser"],
    )

    resume_analyzer_task = Task(
        description=(
            "Analyze the candidate resume below.\n\n"
            "Resume:\n{resume_text}\n\n"
            "Return these sections:\n"
            "1. Candidate headline\n"
            "2. Core skills\n"
            "3. Experience summary\n"
            "4. Education/certifications\n"
            "5. Strongest matches for the role\n"
            "6. Potential concerns or unclear areas"
        ),
        expected_output="A concise candidate profile summary extracted from the resume.",
        agent=agents["resume_analyzer"],
    )

    ats_gap_task = Task(
        description=(
            "Compare the outputs from the JD Parser and Resume Analyzer.\n"
            "Identify ATS gaps, keyword misses, missing experience, and alignment strengths.\n"
            "Provide:\n"
            "1. Match score out of 100\n"
            "2. Missing keywords\n"
            "3. Missing skills or experience\n"
            "4. Existing strengths\n"
            "5. High-impact fixes"
        ),
        expected_output="An ATS gap analysis with a score and actionable improvement points.",
        agent=agents["ats_gap_finder"],
        context=[jd_parser_task, resume_analyzer_task],
    )

    resume_rewriter_task = Task(
        description=(
            "Using the JD summary, resume analysis, and ATS gap analysis, rewrite the resume content.\n"
            "Do not add fake achievements, fake tools, or fake experience.\n"
            "Focus on stronger wording, better keyword alignment, and clearer bullet points.\n"
            "Return:\n"
            "1. Professional summary\n"
            "2. Updated skills section\n"
            "3. Rewritten experience bullets\n"
            "4. Extra tailoring tips"
        ),
        expected_output="A truth-preserving rewritten version of the resume content aligned to the job description.",
        agent=agents["resume_rewriter"],
        context=[jd_parser_task, resume_analyzer_task, ats_gap_task],
    )

    interview_questions_task = Task(
        description=(
            "Using the job description summary, resume analysis, and ATS gap analysis, generate interview questions.\n"
            "Create:\n"
            "1. 5 technical questions\n"
            "2. 5 behavioral questions\n"
            "3. 5 gap-probing questions\n"
            "4. 3 questions the candidate can ask the interviewer"
        ),
        expected_output="A practical interview question set tailored to the role and the candidate profile.",
        agent=agents["interview_question_generator"],
        context=[jd_parser_task, resume_analyzer_task, ats_gap_task],
    )

    return [
        jd_parser_task,
        resume_analyzer_task,
        ats_gap_task,
        resume_rewriter_task,
        interview_questions_task,
    ]


def build_crew() -> Crew:
    llm = build_llm()
    agents = build_agents(llm)
    tasks = build_tasks(agents)
    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_output(output_dir: Path, result: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "resume_parser_report.md"
    report_path.write_text(str(result), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple CrewAI resume parsing workflow with Ollama.")
    parser.add_argument("--resume", required=True, help="Path to the resume text file.")
    parser.add_argument("--jd", required=True, help="Path to the job description text file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where the final report will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    crew = build_crew()
    result = crew.kickoff(
        inputs={
            "resume_text": read_text(resume_path),
            "job_description": read_text(jd_path),
        }
    )

    report_path = save_output(output_dir, result)
    print(f"Report saved to: {report_path.resolve()}")

