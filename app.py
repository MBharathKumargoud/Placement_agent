
import os
import json
import requests
import uvicorn

from pathlib import Path
from html import escape

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

from pypdf import PdfReader
from docx import Document

from google import genai
from google.genai import types

from langchain_core.runnables import RunnableLambda
from langserve import add_routes


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.6-flash"


# ============================================================
# RESUME EXTRACTION
# ============================================================

def extract_resume_text(file_path: str):

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
        "Please upload a valid PDF, DOCX or TXT file."
    )


# ============================================================
# GITHUB
# ============================================================

def github_get(url):

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


def analyze_github(github_url):

    github_url = github_url.strip().rstrip("/")

    if not github_url.startswith(
        "https://github.com/"
    ):

        raise ValueError(
            "Please enter a valid public GitHub URL."
        )

    username = github_url.split(
        "github.com/"
    )[1].split("/")[0]

    if not username:

        raise ValueError(
            "Could not determine GitHub username."
        )

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

            "name":
                repo.get("name"),

            "description":
                repo.get("description"),

            "language":
                language,

            "stars":
                repo.get(
                    "stargazers_count",
                    0
                ),

            "forks":
                repo.get(
                    "forks_count",
                    0
                ),

            "url":
                repo.get("html_url")
        })

    return {

        "profile": {

            "username":
                profile.get("login"),

            "name":
                profile.get("name"),

            "bio":
                profile.get("bio"),

            "public_repositories":
                profile.get("public_repos"),

            "followers":
                profile.get("followers"),

            "profile_url":
                profile.get("html_url")
        },

        "languages":
            languages,

        "repositories":
            repo_data[:30]
    }


# ============================================================
# CAREER AGENT PROMPT
# ============================================================

SYSTEM_INSTRUCTION = """

You are Career Placement Agent.

You are an expert career coach for students
and early-career candidates.

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

If information is missing, explicitly say
that it is missing.

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
# GEMINI
# ============================================================

def generate_career_analysis(
    resume_text,
    github_url,
    target_role
):

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

    # Resume limit
    resume_for_model = resume_text[:30000]

    # GitHub
    try:

        github_data = analyze_github(
            github_url
        )

    except Exception as error:

        github_data = {
            "error": str(error)
        }

    github_for_model = json.dumps(
        github_data,
        indent=2
    )[:30000]

    # Prompt
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


Create the complete personalized
career roadmap.

Base the analysis ONLY on the supplied
resume, GitHub information and target
job role.

Do not invent information.
"""

    client = genai.Client(
        api_key=os.environ[
            "GOOGLE_API_KEY"
        ].strip()
    )

    response = client.models.generate_content(

        model=MODEL,

        contents=user_prompt,

        config=types.GenerateContentConfig(

            system_instruction=
                SYSTEM_INSTRUCTION,

            temperature=0.4
        )
    )

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text


# ============================================================
# LANGCHAIN RUNNABLE
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
# LANGSERVE BACKEND
# ============================================================

add_routes(
    app,
    career_runnable,
    path="/agent-api"
)


# ============================================================
# SINGLE USER PAGE
# ============================================================

@app.get(
    "/agent/playground/",
    response_class=HTMLResponse
)
async def career_page():

    html = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Career Placement Agent</title>

<style>

body {

    font-family:
        Arial,
        sans-serif;

    background:
        #f5f7fb;

    margin:
        0;

    padding:
        30px;
}

.container {

    max-width:
        1000px;

    margin:
        auto;

    background:
        white;

    padding:
        35px;

    border-radius:
        15px;

    box-shadow:
        0 4px 20px
        rgba(0,0,0,0.08);
}

h1 {

    text-align:
        center;
}

.subtitle {

    text-align:
        center;

    color:
        #666;

    margin-bottom:
        30px;
}

label {

    display:
        block;

    font-weight:
        bold;

    margin-top:
        20px;

    margin-bottom:
        8px;
}

input[type="file"],
input[type="text"] {

    width:
        100%;

    padding:
        12px;

    border:
        1px solid #ccc;

    border-radius:
        8px;

    box-sizing:
        border-box;
}

button {

    width:
        100%;

    margin-top:
        25px;

    padding:
        14px;

    border:
        none;

    border-radius:
        8px;

    background:
        #2563eb;

    color:
        white;

    font-size:
        16px;

    cursor:
        pointer;
}

button:hover {

    background:
        #1d4ed8;
}

#status {

    margin-top:
        20px;

    text-align:
        center;

    font-weight:
        bold;
}

#result {

    margin-top:
        30px;

    padding:
        25px;

    background:
        #fafafa;

    border-radius:
        10px;

    white-space:
        pre-wrap;

    line-height:
        1.6;
}

.error {

    color:
        #dc2626;
}

</style>

</head>


<body>

<div class="container">

<h1>
🚀 Career Placement Agent
</h1>

<div class="subtitle">

Upload your resume, provide your GitHub
profile and select your target job role.

</div>


<form id="careerForm">


<label>
📄 Upload Resume
</label>

<input
    type="file"
    id="resume"
    name="resume"
    accept=".pdf,.docx,.txt"
    required
>


<label>
🔗 GitHub Profile URL
</label>

<input
    type="text"
    id="github_url"
    name="github_url"
    placeholder="https://github.com/username"
    required
>


<label>
🎯 Interested Job Role
</label>

<input
    type="text"
    id="target_role"
    name="target_role"
    placeholder="Example: Data Analyst"
    required
>


<button type="submit">

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
            document.getElementById(
                "resume"
            ).files[0];

        const github =
            document.getElementById(
                "github_url"
            ).value;

        const role =
            document.getElementById(
                "target_role"
            ).value;

        const status =
            document.getElementById(
                "status"
            );

        const result =
            document.getElementById(
                "result"
            );


        if (!resume) {

            status.innerHTML =
                '<span class="error">' +
                'Please upload your resume.' +
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
            "⏳ Analyzing your profile...";


        result.innerHTML = "";


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
                    data.detail ||
                    "Backend returned no career analysis."
                );
            }


            status.innerText =
                "✅ Career analysis completed.";


            result.innerHTML =
                data.analysis
                .replace(
                    /\\n/g,
                    "<br>"
                );


        }

        catch(error) {

            status.innerHTML =
                '<span class="error">' +
                "❌ " +
                error.message +
                "</span>";

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

    target_role: str = Form(...)
):

    # Validate extension

    filename = resume.filename or ""

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in [
        ".pdf",
        ".docx",
        ".txt"
    ]:

        return JSONResponse(
            status_code=400,
            content={
                "detail":
                    "Only PDF, DOCX and TXT "
                    "files are supported."
            }
        )


    # Save temporary file

    temp_path = (
        f"/tmp/career_resume"
        f"{extension}"
    )


    file_bytes = await resume.read()


    with open(
        temp_path,
        "wb"
    ) as file:

        file.write(
            file_bytes
        )


    try:

        # Extract resume
        resume_text = extract_resume_text(
            temp_path
        )

        # Generate analysis
        analysis = generate_career_analysis(

            resume_text,

            github_url,

            target_role
        )

        return {
            "analysis":
                analysis
        }

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "detail":
                    str(error)
            }
        )


    finally:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )


# ============================================================
# UVICORN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port
    )
