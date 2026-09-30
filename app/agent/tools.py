"""The three tools, their specs, and one dispatcher that both the agent and the workflow call.

Each description names exactly one job and says what the tool does NOT hold, so no
two descriptions claim the same data:
  get_employee_record    -> facts about one person      (no rules, no figures)
  search_handbook        -> company-wide policy text    (no people, no local rules)
  get_jurisdiction_rules -> one local override, by enum (no free text, no people)
"""

import json
import os
from datetime import date
from typing import Any, Callable, Dict

from app.agent import handbook

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "hr_agent")

# Pinned so tenure (and therefore the expected answers) does not drift day to day.
AS_OF = date.fromisoformat(os.environ.get("HR_AGENT_AS_OF", "2026-09-29"))

JURISDICTIONS = ["US-CA", "US-NY", "UK", "DE", "IN-KA"]
TOPICS = ["notice_period", "annual_leave", "sick_leave", "parental_leave"]

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_record",
            "description": (
                "Return one employee's HR record by employee ID: jurisdiction, role level, employment type, "
                "start date, tenure, leave taken, and links to other records (prior service, transfers). "
                "When a fact depends on a linked record, call this again with that record's ID. "
                "Holds facts about the person only - no policy rules and no entitlement figures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "pattern": "^E\\d{4}$", "description": "e.g. E1004"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_handbook",
            "description": (
                "Keyword-search the company-wide HR handbook: the default policies that apply in every location "
                "(PTO accrual and carry-over, sick days, parental leave, resignation notice, benefits, expenses). "
                "Returns matching sections with section_id and effective date. "
                "Holds no employee data and no jurisdiction's own figures - one country's or state's number "
                "is get_jurisdiction_rules' job, not this tool's."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "3-8 keywords, e.g. 'resignation notice manager'"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_jurisdiction_rules",
            "description": (
                "Return one jurisdiction's own rule for one topic - its PTO days per year, notice by tenure, "
                "sick pay or parental leave - wherever it replaces or adds to the company-wide handbook. "
                "Call once per jurisdiction you need a figure for. "
                "Exact lookup by enum - no free-text search, no employee data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jurisdiction": {"type": "string", "enum": JURISDICTIONS},
                    "topic": {"type": "string", "enum": TOPICS},
                },
                "required": ["jurisdiction", "topic"],
                "additionalProperties": False,
            },
        },
    },
]


def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _service(start: date, end: date) -> Dict[str, Any]:
    years = end.year - start.year - ((end.month, end.day) < (start.month, start.day))
    months = (end.year - start.year) * 12 + end.month - start.month - (end.day < start.day)
    return {"tenure_years": round((end - start).days / 365.25, 2),
            "complete_years_of_service": years, "complete_months_of_service": months}


def get_employee_record(employee_id: str) -> Dict[str, Any]:
    for rec in _load("employees.json")["records"]:
        if rec["employee_id"] == employee_id:
            out = dict(rec)
            end = date.fromisoformat(rec["end_date"]) if rec.get("end_date") else AS_OF
            out["as_of"] = AS_OF.isoformat()
            cont = rec.get("service_continuity") or {}
            if cont.get("counts_toward_tenure"):
                # This tool never follows links (one job), and a tenure computed from this record
                # alone would be a wrong fact - so it says what is missing instead of guessing.
                out.update(tenure_years=None, complete_years_of_service=None, complete_months_of_service=None,
                           tenure_note=f"Not computable from this record: continuous service starts at prior "
                                       f"record {cont['prior_employee_id']}'s start_date.")
            else:
                out.update(_service(date.fromisoformat(rec["start_date"]), end))
            return out
    return {"error": f"no employee record with id {employee_id!r}"}


def search_handbook(query: str) -> Dict[str, Any]:
    hits = handbook.search(query, top_k=3)
    return {"query": query, "sections": hits} if hits else {"query": query, "sections": [], "note": "no match"}


def get_jurisdiction_rules(jurisdiction: str, topic: str) -> Dict[str, Any]:
    if jurisdiction not in JURISDICTIONS:
        return {"error": f"jurisdiction must be one of {JURISDICTIONS}"}
    if topic not in TOPICS:
        return {"error": f"topic must be one of {TOPICS}"}
    doc = _load("jurisdiction_rules.json")
    return {"rule_id": f"JUR-{jurisdiction}-{topic}", "jurisdiction": jurisdiction, "topic": topic,
            "document": doc["document"], "effective": doc["effective"],
            "rule": doc["rules"][jurisdiction][topic]}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_employee_record": get_employee_record,
    "search_handbook": search_handbook,
    "get_jurisdiction_rules": get_jurisdiction_rules,
}


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call. Bad names/arguments come back as an error result, never an exception,
    so a malformed call costs the agent one lap instead of crashing the run."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(TOOLS)}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
