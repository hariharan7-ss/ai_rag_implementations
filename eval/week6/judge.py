"""LLM-as-judge on ONE binary criterion: is the stated policy correct?

The judge will not run until labels_25.json is committed to git and unchanged
since that commit. That is the blind protocol, enforced rather than promised: the
human labels provably exist before the first verdict does.
"""

import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LABELS_PATH = os.path.join(HERE, "labels_25.json")

# A different, larger model than the app's generator (openai/gpt-oss-20b), so the
# judge is not grading its own sampling habits. temperature 0: a judge should not
# flip its verdict between runs.
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_PARAMS = {"temperature": 0, "max_tokens": 2048, "seed": 20260923}

VERDICT_RE = re.compile(r"VERDICT:\s*\**\s*(PASS|FAIL)", re.I)


class LabelsNotCommitted(RuntimeError):
    pass


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def labels_commit():
    """(commit_hash, commit_iso_time) of the commit that added labels_25.json.

    Raises LabelsNotCommitted unless the file exists, is committed, and has not
    been edited since (editing labels after seeing the judge is moving the ruler).
    """
    rel = os.path.relpath(LABELS_PATH, ROOT).replace(os.sep, "/")
    if not os.path.exists(LABELS_PATH):
        raise LabelsNotCommitted(f"{rel} does not exist. Hand-label first: python eval/week6/label.py")
    first = _git("log", "--diff-filter=A", "--format=%H %cI", "--", rel).splitlines()
    if not first:
        raise LabelsNotCommitted(f"{rel} is not committed. Commit it BEFORE running the judge.")
    if _git("status", "--porcelain", "--", rel):
        raise LabelsNotCommitted(f"{rel} has uncommitted changes. Labels may not change once committed.")
    commit, when = first[-1].split(" ", 1)
    return commit, when


def load_prompt(version):
    with open(os.path.join(HERE, f"judge_{version}.txt"), encoding="utf-8") as f:
        return f.read()


_client = None


def _groq():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq()
    return _client


def judge(case, answer, version):
    """{"verdict": "PASS"|"FAIL"|None, "reason": str, "raw": str}"""
    prompt = load_prompt(version)
    prompt = (prompt.replace("{question}", case["question"])
                    .replace("{reference}", case["reference"])
                    .replace("{answer}", answer or ""))
    raw = ""
    for attempt in range(6):
        try:
            resp = _groq().chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], **JUDGE_PARAMS
            )
            raw = resp.choices[0].message.content or ""
            break
        except Exception as e:  # rate limits: back off and retry
            raw = f"ERROR: {e}"
            time.sleep(5 * (attempt + 1))
    m = VERDICT_RE.search(raw)
    verdict = m.group(1).upper() if m else None
    reason = raw[: m.start()].strip() if m else raw.strip()
    return {"verdict": verdict, "reason": reason, "raw": raw}
