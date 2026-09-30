"""Prompt text and output contract shared by the agent and the workflow.

Both systems get the same policy rules and the same contract text, and their
final message goes through the same parser, so "same output contract" is
enforced in code rather than promised.
"""

import json
import re
from typing import Any, Dict, Optional

from app.agent.tools import AS_OF

PROMPT_VERSION = "hr-agent-v1"

POLICY_RULES = f"""Today's date is {AS_OF.isoformat()}.
- A jurisdiction rule overrides the company-wide handbook wherever the two differ.
- If two handbook sections give different figures, use the one with the later effective date.
- Tenure means complete years (or months) of continuous service."""

OUTPUT_CONTRACT = """Reply with ONE JSON object and nothing else:
{"employee_id": "<the employee asked about>",
 "status": "answered" or "cannot_answer",
 "value": <number, or null when cannot_answer>,
 "unit": "days" or "weeks" or "months" or null,
 "answer": "<one sentence>",
 "sources": ["<every rule_id / section_id the figure relies on, copied exactly>"]}
Use "cannot_answer" when the facts you have do not determine the figure."""

AGENT_SYSTEM = f"""You are the Solstice Analytics HR entitlement assistant. Answer the question about one employee's entitlement.
Use the tools to get every fact. Never state a figure you did not read in a tool result or compute from one.
Tool results can point to further records or rules you need - fetch those before answering.
{POLICY_RULES}

When you have the answer, stop calling tools. {OUTPUT_CONTRACT}"""

WORKFLOW_ROUTER = f"""Classify an HR entitlement question for a lookup pipeline.
Return JSON: {{"topic": one of ["notice_period", "annual_leave", "sick_leave", "parental_leave"],
"handbook_query": "<3-8 keywords to search the company-wide HR handbook>"}}"""

WORKFLOW_ANSWER = f"""You are the Solstice Analytics HR entitlement assistant. Answer the question about one employee's entitlement
using only the facts below. Never state a figure that is not in the facts or computed from them.
{POLICY_RULES}

{OUTPUT_CONTRACT}"""

UNITS = {"day": "days", "days": "days", "week": "weeks", "weeks": "weeks", "month": "months", "months": "months"}
EMPLOYEE_ID = re.compile(r"\bE\d{4}\b")


def parse_contract(text: Optional[str]) -> Dict[str, Any]:
    """Model text -> contract dict, or {"contract_error": ...}. Tolerates a ```json fence, nothing else."""
    if not text:
        return {"contract_error": "empty final message"}
    s = text.strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < start:
        return {"contract_error": "no JSON object in final message", "raw": text[:300]}
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError as e:
        return {"contract_error": f"invalid JSON: {e}", "raw": text[:300]}
    if obj.get("status") not in ("answered", "cannot_answer"):
        return {"contract_error": f"bad status {obj.get('status')!r}", "raw": text[:300]}
    value = obj.get("value")
    try:
        value = None if value is None else float(value)
    except (TypeError, ValueError):
        return {"contract_error": f"non-numeric value {value!r}", "raw": text[:300]}
    unit = obj.get("unit")
    unit = UNITS.get(str(unit).strip().lower(), unit) if unit is not None else None
    sources = obj.get("sources") or []
    if not isinstance(sources, list):
        sources = [sources]
    return {"employee_id": obj.get("employee_id"), "status": obj["status"], "value": value, "unit": unit,
            "answer": obj.get("answer", ""), "sources": [str(x) for x in sources]}


def stopped_contract(question: str, reason: str) -> Dict[str, Any]:
    """The well-formed answer returned when a run is stopped (budget hit, pipeline error)."""
    m = EMPLOYEE_ID.search(question)
    return {"employee_id": m.group(0) if m else None, "status": "cannot_answer", "value": None,
            "unit": None, "answer": reason, "sources": []}
