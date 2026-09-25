import os
import json
import time
import tempfile
import traceback
from pathlib import Path
from urllib.parse import urlparse

import requests
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from pypdf import PdfReader
from docx import Document

from google import genai
from google.genai import types

from langchain_core.runnables import RunnableLambda
from langserve import add_routes


# ============================================================
# CONFIGURATION
# ============================================================

# Both are stable Gemini models.
# You can override these in Render Environment Variables.
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")

MAX_RESUME_CHARS = 30000
MAX_GITHUB_CHARS = 30000

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ============================================================
# RESUME EXTRACTION
# ============================================================

def extract_resume_text(file_path: str):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(file_path)
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        text = "\n".join(pages)

        if text.strip():
            return text

    elif extension == ".docx":
        document = Document(file_path)
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        if text.strip():
            return text

    elif extension == ".txt":
        text = Path(file_path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if text.strip():
            return text

    raise ValueError(
        "Could not extract text from the resume. "
        "Please upload a valid PDF, DOCX or TXT file."
    )


# ============================================================
# GITHUB
# ============================================================

def github_get(url: str):
    response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "CareerPlacementAgent"
        },
        timeout=20
    )
    response.raise_for_status()
    return response.json()


def analyze_github(github_url: str):
    github_url = github_url.strip().rstrip("/")
    parsed = urlparse(github_url)

    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
    ):
        raise ValueError(
            "Please enter a valid public GitHub URL."
        )

    parts = [p for p in parsed.path.split("/") if p]

    if not parts:
        raise ValueError(
            "Could not determine GitHub username."
        )

    username = parts[0]

    profile = github_get(
        f"https://api.github.com/users/{username}"
    )

    repositories = github_get(
        f"https://api.github.com/users/{username}/repos"
        "?per_page=100&sort=updated"
    )

    repo_data = []
    languages = {}

    for repo in repositories:
        if repo.get("fork"):
            continue

        language = repo.get("language")

        if language:
            languages[language] = (
                languages.get(language, 0) + 1
            )

        repo_data.append({
            "name": repo.get("name"),
            "description": repo.get("description"),
            "language": language,
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "url": repo.get("html_url")
        })

    return {
        "profile": {
            "username": profile.get("login"),
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "public_repositories": profile.get("public_repos"),
            "followers": profile.get("followers"),
            "profile_url": profile.get("html_url")
        },
        "languages": languages,
        "repositories": repo_data[:30]
    }


# ============================================================
# CAREER AGENT PROMPT
# ============================================================

SYSTEM_INSTRUCTION = """
You are Career Placement Agent.

You are an expert career coach for students and early-career candidates.

Analyze exactly THREE sources:
1. Resume
2. Public GitHub profile/repositories
3. Candidate's target job role

Never invent:
- Experience
- Skills
- Projects
- Certifications
- Achievements
- GitHub activity
- Job offers

If information is missing, explicitly say that it is missing.

Your final answer MUST contain:

# 1. Candidate Snapshot

# 2. Target Role Fit Score
Give a score out of 100 with explanation.

# 3. Strong Skills Already Present

# 4. Missing / Weak Skills
For every missing skill explain:
- WHY it matters
- HOW to learn it

# 5. Resume Improvements

# 6. GitHub Improvements
Give concrete actions.

# 7. Skills to Learn in Priority Order

# 8. Recommended Projects
Include:
- Beginner
- Intermediate
- Portfolio-level

For each project explain:
- Objective
- Technologies
- Features
- Learning outcome
- Recruiter value

# 9. Interview Preparation
Include:
- Technical
- SQL/coding when relevant
- Behavioral
- Project questions

# 10. 30-Day Action Plan
Give weekly goals.

# 11. 60-90 Day Roadmap

# 12. Suggested Job Titles to Search For

# 13. Final Recommendation
Give the single most important next step.

Use clear Markdown.
Use tables and checklists where useful.
Never invent information.
"""


# ============================================================
# GEMINI ERROR DETECTION
# ============================================================

def is_retryable_error(error):
    text = str(error).upper()
    code = getattr(error, "code", None)

    return (
        code in {408, 429, 500, 502, 503, 504}
        or "503" in text
        or "UNAVAILABLE" in text
        or "SERVICE_UNAVAILABLE" in text
        or "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or "TIMEOUT" in text
    )


# ============================================================
# GEMINI CALL WITH RETRY + FALLBACK
# ============================================================

def call_gemini_with_resilience(client, prompt):
    models_to_try = [PRIMARY_MODEL]

    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None

    for model_name in models_to_try:

        # 3 attempts per model.
        for attempt in range(3):

            try:
                print(
                    f"Gemini model={model_name}, "
                    f"attempt={attempt + 1}/3"
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                )

                if response and response.text:
                    print(
                        f"Gemini success using {model_name}"
                    )
                    return response.text

                raise RuntimeError(
                    f"{model_name} returned an empty response."
                )

            except Exception as error:
                last_error = error

                print(
                    f"Gemini error on {model_name}: "
                    f"{error}"
                )

                if not is_retryable_error(error):
                    raise

                if attempt < 2:
                    # 2, 4 seconds, then move to fallback.
                    delay = 2 ** (attempt + 1)

                    print(
                        f"Retrying {model_name} in "
                        f"{delay} seconds..."
                    )

                    time.sleep(delay)

        print(
            f"{model_name} unavailable after retries."
        )

    raise RuntimeError(
        "Gemini is temporarily unavailable. "
        "Both the primary and fallback models "
        "failed after retries. Please try again shortly."
    ) from last_error


# ============================================================
# CAREER ANALYSIS
# ============================================================

def generate_career_analysis(
    resume_text,
    github_url,
    target_role
):
    if not resume_text.strip():
        raise ValueError("Resume could not be read.")

    if not github_url.strip():
        raise ValueError("GitHub URL is required.")

    if not target_role.strip():
        raise ValueError("Target job role is required.")

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is missing in Render Environment Variables."
        )

    resume_for_model = resume_text[:MAX_RESUME_CHARS]

    try:
        github_data = analyze_github(github_url)
    except Exception as error:
        print("GitHub error:", error)
        github_data = {
            "error": str(error)
        }

    github_for_model = json.dumps(
        github_data,
        indent=2
    )[:MAX_GITHUB_CHARS]

    user_prompt = f"""
TARGET JOB ROLE
===============
{target_role}

RESUME
======
{resume_for_model}

PUBLIC GITHUB PROFILE
=====================
{github_for_model}

Create the complete personalized career roadmap.

Base the analysis ONLY on:
1. Supplied resume
2. Supplied GitHub information
3. Target job role

Do not invent information.
If something is unavailable, clearly say that it is missing.
"""

    client = genai.Client(
        api_key=api_key.strip()
    )

    return call_gemini_with_resilience(
        client,
        user_prompt
    )


# ============================================================
# LANGCHAIN
# ============================================================

career_runnable = RunnableLambda(
    lambda x: generate_career_analysis(
        x["resume_text"],
        x["github_url"],
        x["target_role"]
    )
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Career Placement Agent",
    version="1.0"
)


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    career_runnable,
    path="/agent-api"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/agent/playground/",
    response_class=HTMLResponse
)
async def career_page():

    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Career Placement Agent</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: #f5f7fb;
    margin: 0;
    padding: 30px;
}

.container {
    max-width: 1000px;
    margin: auto;
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

label {
    display: block;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 8px;
}

input[type="file"],
input[type="text"] {
    width: 100%;
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-sizing: border-box;
}

button {
    width: 100%;
    margin-top: 25px;
    padding: 14px;
    border: none;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

button:disabled {
    background: #93c5fd;
    cursor: not-allowed;
}

#status {
    margin-top: 20px;
    text-align: center;
    font-weight: bold;
}

#result {
    margin-top: 30px;
    padding: 25px;
    background: #fafafa;
    border-radius: 10px;
    white-space: pre-wrap;
    line-height: 1.6;
    overflow-x: auto;
}

.error {
    color: #dc2626;
}

.success {
    color: #16a34a;
}
</style>
</head>

<body>

<div class="container">

<h1>🚀 Career Placement Agent</h1>

<div class="subtitle">
Upload your resume, provide your GitHub profile and enter your target job role.
</div>

<form id="careerForm">

<label>📄 Upload Resume</label>

<input
    type="file"
    id="resume"
    accept=".pdf,.docx,.txt"
    required
>

<label>🔗 GitHub Profile URL</label>

<input
    type="text"
    id="github_url"
    placeholder="https://github.com/username"
    required
>

<label>🎯 Interested Job Role</label>

<input
    type="text"
    id="target_role"
    placeholder="Example: Data Analyst"
    required
>

<button type="submit" id="analyzeButton">
🚀 Analyze My Career
</button>

</form>

<div id="status"></div>
<div id="result"></div>

</div>

<script>
document
.getElementById("careerForm")
.addEventListener("submit", async function(event) {

    event.preventDefault();

    const resume =
        document.getElementById("resume").files[0];

    const github =
        document.getElementById("github_url").value.trim();

    const role =
        document.getElementById("target_role").value.trim();

    const status =
        document.getElementById("status");

    const result =
        document.getElementById("result");

    const button =
        document.getElementById("analyzeButton");

    if (!resume) {
        status.innerHTML =
            '<span class="error">Please upload your resume.</span>';
        return;
    }

    if (!github) {
        status.innerHTML =
            '<span class="error">Please enter your GitHub URL.</span>';
        return;
    }

    if (!role) {
        status.innerHTML =
            '<span class="error">Please enter your target job role.</span>';
        return;
    }

    const formData = new FormData();

    formData.append("resume", resume);
    formData.append("github_url", github);
    formData.append("target_role", role);

    button.disabled = true;
    button.innerText = "⏳ Analyzing...";

    status.innerText =
        "⏳ Analyzing your profile...";

    result.innerText = "";

    try {

        const response = await fetch(
            "/agent/analyze",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Analysis failed."
            );
        }

        status.innerHTML =
            '<span class="success">✅ Career analysis completed.</span>';

        result.textContent =
            data.analysis || "";

    } catch (error) {

        console.error(error);

        status.innerHTML =
            '<span class="error">❌ ' +
            error.message +
            '</span>';

    } finally {

        button.disabled = false;
        button.innerText = "🚀 Analyze My Career";
    }
});
</script>

</body>
</html>
"""

    return HTMLResponse(content=html)


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/agent/analyze")
async def analyze_uploaded_resume(
    resume: UploadFile = File(...),
    github_url: str = Form(...),
    target_role: str = Form(...)
):
    filename = resume.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files are supported."
        )

    file_bytes = await resume.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded resume is empty."
        )

    temp_path = None

    try:
        fd, temp_path = tempfile.mkstemp(
            suffix=extension
        )
        os.close(fd)

        with open(temp_path, "wb") as file:
            file.write(file_bytes)

        resume_text = extract_resume_text(
            temp_path
        )

        analysis = generate_career_analysis(
            resume_text,
            github_url,
            target_role
        )

        return {
            "analysis": analysis
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Career analysis failed: {error}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", "8000")
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
