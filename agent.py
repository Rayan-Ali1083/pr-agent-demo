"""Autonomous PR review agent: LangGraph + Gemini (langchain-google-genai).

Flow: fetch_context -> review -> (generate_fix -> run_tests -> push | retry | comment)
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

API = "https://api.github.com"
MAX_ATTEMPTS = 2
MAX_FILE_CHARS = 20000
REPO_ROOT = Path.cwd().resolve()


# ---------- State ----------
class AgentState(TypedDict, total=False):
    repo: str
    pr_number: int
    head_ref: str
    diff: str
    changed_files: List[str]
    file_contents: Dict[str, str]
    repo_tree: str
    issues: List[dict]
    fixes: Dict[str, str]
    test_output: str
    tests_passed: bool
    attempts: int
    result: str


# ---------- Structured output schemas ----------
class Issue(BaseModel):
    file: str
    line: Optional[int] = None
    severity: str = Field(description="low | medium | high | critical")
    category: str = Field(description="logic | security")
    description: str


class ReviewResult(BaseModel):
    issues: List[Issue] = Field(default_factory=list)


class FileFix(BaseModel):
    path: str = Field(description="Repo-relative path of an existing .py file")
    content: str = Field(description="Complete new file content")


class FixResult(BaseModel):
    files: List[FileFix]


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0,
    )


def gh_headers(accept: str = "application/vnd.github+json") -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": accept,
    }


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def safe_path(rel: str) -> Optional[Path]:
    """Allow only existing .py files inside the repo, outside .github."""
    p = (REPO_ROOT / rel).resolve()
    if REPO_ROOT not in p.parents or p.suffix != ".py" or not p.exists():
        return None
    if ".github" in p.relative_to(REPO_ROOT).parts:
        return None
    return p


# ---------- Nodes ----------
def fetch_context(state: AgentState) -> AgentState:
    repo, pr = state["repo"], state["pr_number"]
    diff = requests.get(
        f"{API}/repos/{repo}/pulls/{pr}",
        headers=gh_headers("application/vnd.github.v3.diff"),
        timeout=30,
    )
    diff.raise_for_status()

    files = requests.get(
        f"{API}/repos/{repo}/pulls/{pr}/files?per_page=100",
        headers=gh_headers(),
        timeout=30,
    )
    files.raise_for_status()
    changed = [
        f["filename"] for f in files.json()
        if f["filename"].endswith(".py") and f["status"] != "removed"
    ]

    contents = {}
    for name in changed:
        p = REPO_ROOT / name
        if p.exists():
            contents[name] = p.read_text()[:MAX_FILE_CHARS]

    tree = run(["git", "ls-files"]).stdout
    return {
        "diff": diff.text[:60000],
        "changed_files": changed,
        "file_contents": contents,
        "repo_tree": tree[:5000],
        "attempts": 0,
    }


def review(state: AgentState) -> AgentState:
    if not state["changed_files"]:
        return {"issues": [], "result": "No Python files changed."}

    files_block = "\n\n".join(
        f"=== {n} ===\n{c}" for n, c in state["file_contents"].items()
    )
    prompt = f"""You are a senior code reviewer. Find real logic bugs and security
vulnerabilities (unhandled edge cases, injection, unsafe deserialization, secrets,
auth flaws, etc.) introduced or exposed by this pull request. Ignore style.
Return an empty list if the code is correct.

Repository files:
{state['repo_tree']}

PR diff:
{state['diff']}

Full contents of changed files:
{files_block}
"""
    llm = get_llm().with_structured_output(ReviewResult)
    out: ReviewResult = llm.invoke(prompt)
    return {"issues": [i.model_dump() for i in out.issues]}


def generate_fix(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    # Re-read current file state (previous attempt reverted on failure)
    current = {
        n: (REPO_ROOT / n).read_text()[:MAX_FILE_CHARS]
        for n in state["changed_files"] if (REPO_ROOT / n).exists()
    }
    files_block = "\n\n".join(f"=== {n} ===\n{c}" for n, c in current.items())
    feedback = ""
    if state.get("test_output") and attempts > 1:
        feedback = f"\nYour previous fix failed the tests:\n{state['test_output'][-3000:]}\n"

    prompt = f"""Fix the following issues with minimal changes. Return the COMPLETE
new content of every file you modify. Do not modify test files unless the test
itself is wrong. Keep existing public APIs.

Issues:
{state['issues']}
{feedback}
Files:
{files_block}
"""
    llm = get_llm().with_structured_output(FixResult)
    out: FixResult = llm.invoke(prompt)

    fixes = {}
    for f in out.files:
        p = safe_path(f.path)
        if p is None:
            print(f"Skipping disallowed path: {f.path}")
            continue
        p.write_text(f.content)
        fixes[f.path] = f.content
    return {"fixes": fixes, "attempts": attempts}


def run_tests(state: AgentState) -> AgentState:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=600,
    )
    output = (proc.stdout + proc.stderr)[-4000:]
    return {"tests_passed": proc.returncode == 0 and bool(state.get("fixes")),
            "test_output": output}


def revert_changes(state: AgentState) -> AgentState:
    run(["git", "checkout", "--", "."])
    return {}


def push_commit(state: AgentState) -> AgentState:
    run(["git", "config", "user.name", "pr-review-agent[bot]"])
    run(["git", "config", "user.email", "pr-review-agent@users.noreply.github.com"])
    run(["git", "add", *state["fixes"].keys()])
    msg = "fix: automated review fixes\n\n" + "\n".join(
        f"- [{i['severity']}] {i['file']}: {i['description']}" for i in state["issues"]
    )
    commit = run(["git", "commit", "-m", msg])
    if commit.returncode != 0:
        return {"result": f"Commit failed: {commit.stderr}"}
    push = run(["git", "push", "origin", f"HEAD:{state['head_ref']}"])
    if push.returncode != 0:
        return {"result": f"Push failed: {push.stderr}"}
    _comment(state, "Automated review found issues and pushed a tested fix:\n\n"
             + _format_issues(state["issues"]))
    return {"result": "Fix pushed."}


def post_comment(state: AgentState) -> AgentState:
    body = ("Automated review found issues but could not produce a fix that passes tests.\n\n"
            + _format_issues(state["issues"])
            + f"\n\nLast test output:\n```\n{state.get('test_output', '')}\n```")
    _comment(state, body)
    return {"result": "Comment posted (fix failed tests)."}


# ---------- Helpers ----------
def _format_issues(issues: List[dict]) -> str:
    return "\n".join(
        f"- **{i['severity']}** ({i['category']}) `{i['file']}`"
        f"{':' + str(i['line']) if i.get('line') else ''}: {i['description']}"
        for i in issues
    )


def _comment(state: AgentState, body: str) -> None:
    r = requests.post(
        f"{API}/repos/{state['repo']}/issues/{state['pr_number']}/comments",
        headers=gh_headers(), json={"body": body}, timeout=30,
    )
    r.raise_for_status()


# ---------- Routing ----------
def after_review(state: AgentState) -> str:
    return "generate_fix" if state.get("issues") else END


def after_tests(state: AgentState) -> str:
    if state["tests_passed"]:
        return "push_commit"
    if state["attempts"] < MAX_ATTEMPTS:
        return "revert_changes"
    return "revert_and_comment"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("fetch_context", fetch_context)
    g.add_node("review", review)
    g.add_node("generate_fix", generate_fix)
    g.add_node("run_tests", run_tests)
    g.add_node("revert_changes", revert_changes)
    g.add_node("revert_and_comment", lambda s: revert_changes(s))
    g.add_node("push_commit", push_commit)
    g.add_node("post_comment", post_comment)

    g.set_entry_point("fetch_context")
    g.add_edge("fetch_context", "review")
    g.add_conditional_edges("review", after_review, {"generate_fix": "generate_fix", END: END})
    g.add_edge("generate_fix", "run_tests")
    g.add_conditional_edges("run_tests", after_tests, {
        "push_commit": "push_commit",
        "revert_changes": "revert_changes",
        "revert_and_comment": "revert_and_comment",
    })
    g.add_edge("revert_changes", "generate_fix")  # retry with test feedback
    g.add_edge("revert_and_comment", "post_comment")
    g.add_edge("push_commit", END)
    g.add_edge("post_comment", END)
    return g.compile()


if __name__ == "__main__":
    final = build_graph().invoke({
        "repo": os.environ["REPO"],
        "pr_number": int(os.environ["PR_NUMBER"]),
        "head_ref": os.environ["HEAD_REF"],
    })
    print(final.get("result", "Done."))
