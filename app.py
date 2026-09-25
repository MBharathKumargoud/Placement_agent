import os
import json
import time
import tempfile
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

# Models are tried in this order.
# If one model is temporarily unavailable, the next one is tried.
# Start with the lighter, high-throughput models.
# Google currently lists these as stable Gemini 3 models.
# Flash-Lite is a better fit for a high-volume career-analysis app.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]

# The Gemini Python SDK already performs automatic retries for
# transient 429/5xx errors. These are additional model-level
# fallbacks if the first model remains unavailable.
MAX_RETRIES_PER_MODEL = 2

# Extra wait after the SDK has returned a transient error.
# This prevents all fallback requests from hitting the service
# at the same moment.
RETRY_DELAYS = [5, 15]

# Gemini transient errors that are normally worth retrying.
RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# RESUME EXTRACTION
# ============================================================

def extract_resume_text(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    # PDF
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

    # DOCX
    elif extension == ".docx":
        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        if text.strip():
            return text

    # TXT
    elif extension == ".txt":
        text = Path(file_path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if text.strip():
            return text

    raise ValueError(
        "Could not extract text from the resume. "
        "Please upload a valid PDF, DOCX or TXT file "
        "containing selectable text."
    )


# ============================================================
# GITHUB
# ============================================================

def github_get(url: str):
    response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "CareerPlacementAgent",
        },
        timeout=20,
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
            "Please enter a valid public GitHub profile URL, "
            "for example: https://github.com/username"
        )

    parts = [
        part
        for part in parsed.path.split("/")
        if part
    ]
    if not parts:
        raise ValueError(
            "Could not determine GitHub username."
        )

    username = parts[0]

    if username.lower() in {"login", "signup", "settings"}:
        raise ValueError(
            "Please enter a GitHub profile URL, not a GitHub site page."
        )

    print(f"[GitHub] Analyzing profile: {username}", flush=True)

    # IMPORTANT:
    # These are real API URLs. Do not replace them with
    # Markdown/HTML links.
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

        repo_data.append(
            {
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": language,
                "stars": repo.get(
                    "stargazers_count",
                    0,
                ),
                "forks": repo.get(
                    "forks_count",
                    0,
                ),
                "url": repo.get("html_url"),
            }
        )

    return {
        "profile": {
            "username": profile.get("login"),
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "public_repositories": profile.get(
                "public_repos"
            ),
            "followers": profile.get("followers"),
            "profile_url": profile.get("html_url"),
        },
        "languages": languages,
        "repositories": repo_data[:30],
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
# GEMINI ERROR HELPERS
# ============================================================

def error_text(error: Exception) -> str:
    """
    Convert an exception into a useful log/frontend message
    without exposing the API key.
    """
    message = str(error).strip()

    if not message:
        message = repr(error)

    # Never accidentally display the API key if an SDK exception
    # contains it in its text.
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if api_key and api_key in message:
        message = message.replace(
            api_key,
            "[REDACTED_API_KEY]"
        )

    return message


def is_retryable_error(error: Exception) -> bool:
    """
    Detect common transient Gemini/API errors.
    """
    message = error_text(error).lower()

    for code in RETRYABLE_STATUS_CODES:
        if str(code) in message:
            return True

    transient_words = [
        "unavailable",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "temporarily",
        "internal server error",
        "deadline exceeded",
        "timeout",
        "timed out",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    ]

    return any(
        word in message
        for word in transient_words
    )


# ============================================================
# GEMINI CAREER ANALYSIS
# ============================================================

def generate_career_analysis(
    resume_text: str,
    github_url: str,
    target_role: str,
) -> str:

    if not resume_text.strip():
        raise ValueError(
            "Resume could not be read."
        )

    if not github_url.strip():
        raise ValueError(
            "GitHub URL is required."
        )

    if not target_role.strip():
        raise ValueError(
            "Target job role is required."
        )

    google_api_key = os.getenv(
        "GOOGLE_API_KEY",
        ""
    ).strip()

    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured on the server."
        )

    # Keep the request size reasonable.
    resume_for_model = resume_text[:30000]

    # --------------------------------------------------------
    # GitHub
    # --------------------------------------------------------

    try:
        github_data = analyze_github(github_url)

    except Exception as error:
        github_error = error_text(error)

        print(
            f"[GitHub ERROR] {github_error}",
            flush=True
        )

        # Do not completely stop the career analysis just because
        # GitHub could not be read.
        github_data = {
            "error": (
                "GitHub data could not be retrieved. "
                f"Reason: {github_error}"
            )
        }

    github_for_model = json.dumps(
        github_data,
        indent=2,
        ensure_ascii=False,
    )[:30000]

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

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
1. The supplied resume
2. The supplied public GitHub information
3. The supplied target job role

Do not invent information.
If information is unavailable, explicitly state that it is unavailable.
"""

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    client = genai.Client(
        api_key=google_api_key
    )

    errors = []

    # --------------------------------------------------------
    # Try all configured models
    # --------------------------------------------------------

    for model_name in MODELS:

        for attempt in range(
            1,
            MAX_RETRIES_PER_MODEL + 1
        ):

            print(
                f"[Gemini] model={model_name} "
                f"attempt={attempt}",
                flush=True,
            )

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.4,
                    ),
                )

                if not response.text:
                    raise ValueError(
                        f"{model_name} returned an empty response."
                    )

                print(
                    f"[Gemini SUCCESS] model={model_name}",
                    flush=True,
                )

                return response.text

            except Exception as error:

                message = error_text(error)

                print(
                    f"[Gemini ERROR] "
                    f"model={model_name} "
                    f"attempt={attempt}: "
                    f"{message}",
                    flush=True,
                )

                errors.append(
                    f"{model_name} attempt {attempt}: {message}"
                )

                # Non-transient errors should immediately move
                # to the next model.
                if not is_retryable_error(error):
                    print(
                        f"[Gemini] Non-retryable error for "
                        f"{model_name}; moving to next model.",
                        flush=True,
                    )
                    break

                # Retry transient errors only.
                if attempt < MAX_RETRIES_PER_MODEL:
                    wait_seconds = RETRY_DELAYS[
                        min(
                            attempt - 1,
                            len(RETRY_DELAYS) - 1
                        )
                    ]

                    print(
                        f"[Gemini] Retrying "
                        f"{model_name} in "
                        f"{wait_seconds}s...",
                        flush=True,
                    )

                    time.sleep(wait_seconds)

    # --------------------------------------------------------
    # All models failed
    # --------------------------------------------------------

    print(
        "[Gemini FINAL ERROR] All configured models failed.",
        flush=True,
    )

    # Keep the useful information, but cap its size.
    error_summary = "\n".join(errors[-8:])[:5000]

    raise RuntimeError(
        "Gemini request failed after trying all configured models.\n\n"
        f"Attempts:\n{error_summary}"
    )


# ============================================================
# LANGCHAIN RUNNABLE
# ============================================================

career_runnable = RunnableLambda(
    lambda x: generate_career_analysis(
        x["resume_text"],
        x["github_url"],
        x["target_role"],
    )
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Career Placement Agent",
    version="2.0",
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Career Placement Agent",
        "models": MODELS,
        "gemini_key_configured": bool(
            os.getenv("GOOGLE_API_KEY", "").strip()
        ),
    }


# ============================================================
# LANGSERVE BACKEND
# ============================================================

add_routes(
    app,
    career_runnable,
    path="/agent-api",
)


# ============================================================
# SINGLE USER PAGE
# ============================================================

@app.get(
    "/agent/playground/",
    response_class=HTMLResponse,
)
async def career_page():

    html = """
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
    background: #94a3b8;
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
    white-space: pre-wrap;
}

.success {
    color: #15803d;
}

.small {
    color: #666;
    font-size: 13px;
    margin-top: 8px;
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
    name="resume"
    accept=".pdf,.docx,.txt"
    required
>

<div class="small">
Supported formats: PDF, DOCX, TXT
</div>


<label>🔗 GitHub Profile URL</label>

<input
    type="text"
    id="github_url"
    name="github_url"
    placeholder="https://github.com/username"
    required
>


<label>🎯 Interested Job Role</label>

<input
    type="text"
    id="target_role"
    name="target_role"
    placeholder="Example: Data Analyst"
    required
>


<button
    id="analyzeButton"
    type="submit"
>
🚀 Analyze My Career
</button>

</form>


<div id="status"></div>

<div id="result"></div>

</div>


<script>

document
.getElementById("careerForm")
.addEventListener(
    "submit",
    async function(event) {

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
                '<span class="error">' +
                'Please upload your resume.' +
                '</span>';

            return;
        }


        if (!github) {

            status.innerHTML =
                '<span class="error">' +
                'Please enter your GitHub URL.' +
                '</span>';

            return;
        }


        if (!role) {

            status.innerHTML =
                '<span class="error">' +
                'Please enter your target job role.' +
                '</span>';

            return;
        }


        const formData =
            new FormData();

        formData.append(
            "resume",
            resume
        );

        formData.append(
            "github_url",
            github
        );

        formData.append(
            "target_role",
            role
        );


        status.innerText =
            "⏳ Analyzing your profile... " +
            "This may take a little while.";

        result.innerText = "";

        button.disabled = true;
        button.innerText = "⏳ Analyzing...";


        try {

            const response =
                await fetch(
                    "/agent/analyze",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Analysis failed."
                );
            }


            if (!data.analysis) {

                throw new Error(
                    "The server returned an empty analysis."
                );
            }


            status.innerHTML =
                '<span class="success">' +
                '✅ Career analysis completed.' +
                '</span>';


            // textContent keeps the generated response safe.
            // Markdown is shown as readable plain text.
            result.textContent =
                data.analysis;

        }

        catch(error) {

            status.innerHTML =
                '<span class="error">' +
                "❌ " +
                error.message +
                '</span>';

            console.error(
                "Career analysis error:",
                error
            );

        }

        finally {

            button.disabled = false;
            button.innerText =
                "🚀 Analyze My Career";
        }

    }
);

</script>

</body>
</html>
"""

    return HTMLResponse(
        content=html
    )


# ============================================================
# ANALYZE UPLOADED RESUME
# ============================================================

@app.post("/agent/analyze")
async def analyze_uploaded_resume(
    resume: UploadFile = File(...),
    github_url: str = Form(...),
    target_role: str = Form(...),
):

    print(
        "[Request] POST /agent/analyze "
        f"filename={resume.filename!r} "
        f"github={github_url!r} "
        f"role={target_role!r}",
        flush=True,
    )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    filename = resume.filename or ""

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in {
        ".pdf",
        ".docx",
        ".txt",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, DOCX and TXT files "
                "are supported."
            ),
        )


    # --------------------------------------------------------
    # Validate form values
    # --------------------------------------------------------

    github_url = github_url.strip()
    target_role = target_role.strip()

    if not github_url:
        raise HTTPException(
            status_code=400,
            detail="GitHub URL is required.",
        )

    if not target_role:
        raise HTTPException(
            status_code=400,
            detail="Target job role is required.",
        )


    # --------------------------------------------------------
    # Save uploaded resume to a unique temporary file
    # --------------------------------------------------------

    temp_path = None

    try:

        file_bytes = await resume.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded resume is empty.",
            )

        fd, temp_path = tempfile.mkstemp(
            suffix=extension,
            prefix="career_resume_",
        )

        with os.fdopen(fd, "wb") as file:
            file.write(file_bytes)


        # ----------------------------------------------------
        # Extract resume
        # ----------------------------------------------------

        print(
            "[Resume] Extracting text...",
            flush=True,
        )

        resume_text = extract_resume_text(
            temp_path
        )

        print(
            f"[Resume] Extracted "
            f"{len(resume_text)} characters.",
            flush=True,
        )


        # ----------------------------------------------------
        # Generate analysis
        # ----------------------------------------------------

        analysis = generate_career_analysis(
            resume_text,
            github_url,
            target_role,
        )


        print(
            "[Request] Career analysis completed.",
            flush=True,
        )

        return {
            "analysis": analysis
        }


    except HTTPException:
        raise


    except Exception as error:

        message = error_text(error)

        print(
            f"[Request ERROR] {message}",
            flush=True,
        )

        # IMPORTANT:
        # This returns the actual error to the frontend instead
        # of hiding everything behind:
        # "Both primary and fallback models failed."
        raise HTTPException(
            status_code=500,
            detail=message,
        )


    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# UVICORN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
