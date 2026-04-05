# Resume Parser with CrewAI

Simple Python multi-agent resume parsing flow using CrewAI and a local Ollama model.

## Agents

- JD Parser
- Resume Analyzer
- ATS Gap Finder
- Resume Rewriter
- Interview Question Generator

## What it does

The app reads:

- a resume text file
- a job description text file

Then it runs the five agents in sequence and produces a markdown report with:

- parsed job requirements
- resume analysis
- ATS gap analysis
- tailored resume rewrite suggestions
- interview questions

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Make sure Ollama is running locally.
4. Pull a model, for example:

```powershell
ollama pull llama3.1:8b
```

5. Copy `.env.example` to `.env` and adjust if needed.

## Run

Prepare two text files, for example:

- `data/resume.txt`
- `data/job_description.txt`

Run:

```powershell
python main.py --resume data/resume.txt --jd data/job_description.txt
```

The final report will be saved to:

```text
output/resume_parser_report.md
```

## Notes

- Keep the input files as plain text for this simple version.
- The resume rewriter is instructed not to invent experience or achievements.
- You can change the Ollama model through `OLLAMA_MODEL` in `.env`.
