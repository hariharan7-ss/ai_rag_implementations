"""Redact employee identifiers BEFORE anything is written to the trace file.

Called from inside trace_logger.log_trace, in the write path, so identifiers
never reach traces/traces.jsonl at all. This is not a cleanup pass over an
existing file.

Design note (a first attempt was thrown away): a generic "capitalised words are
names" rule cannot be used here. HR policy text is full of capitalised headings
-- Paid Time Off, Review Cycle, Leave Duration -- and redacting those makes the
traces unreadable, which defeats the point of keeping them. So:

  1. Structured identifiers (employee ID, email, phone, SSN, bank/account) are
     matched by regex in every field. High precision, safe on policy text.
  2. Person names are detected ONLY in the employee-supplied query, and only
     after an explicit trigger ("my name is X", "I'm X", "my manager X"). The
     literal names found there are then redacted across the whole record, so an
     answer echoing the name is covered too. Retrieved policy text is never
     name-scanned and passes through intact.

This trades recall for precision: a name with no trigger and no identifier is
not caught. That limitation is recorded in notes.md rather than hidden.
"""

import re
from typing import Any, Dict, List, Tuple

REDACTION_VERSION = "redact-v2"

_STRUCTURED: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:EMP|EMPL|STAFF)[-_ ]?\d{3,8}\b", re.I), "[EMPLOYEE_ID]"),
    (re.compile(r"\bE[-_]\d{4,8}\b", re.I), "[EMPLOYEE_ID]"),
    (re.compile(r"\b(?:employee|staff|payroll|badge)\s*(?:id|number|no\.?|#)?\s*[:# ]\s*\d{3,10}\b", re.I), "[EMPLOYEE_ID]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[BANK_ACCOUNT]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT_NUMBER]"),
]

# A captured name never extends across one of these.
_STOP = {
    "and", "or", "but", "is", "are", "was", "were", "who", "which", "that",
    "can", "do", "does", "did", "has", "have", "had", "will", "would", "my",
    "our", "his", "her", "their", "the", "a", "an", "in", "on", "at", "to",
    "for", "of", "from", "with", "id", "ids", "number", "no", "code", "policy",
    "hr", "it", "pto", "i", "me", "we", "you", "if", "so", "then", "about",
    "leave", "team", "manager", "employee", "here", "today", "now", "please",
}

# Trigger matching is case-insensitive, but the NAME capture stays
# case-sensitive: a global re.I would let it swallow ordinary lowercase words
# ("Ravi Kumar approved"). The scoped (?i:...) group keeps that distinction.
_TRIGGERS = re.compile(
    r"(?i:\bmy name is\b|\bi am\b|\bi'm\b|\bthis is\b|\bspeaking with\b"
    r"|\bon behalf of\b|\bmy manager\b|\bmy supervisor\b|\bmy colleague\b"
    r"|\bmy coworker\b|\bmy report\b|\breports? to\b|\breporting to\b"
    r"|\bmanager,?\b|\bMr\.|\bMrs\.|\bMs\.|\bDr\.)"
    r"\s+(?P<name>[A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15}){0,2})"
)


def _trim(name: str) -> str:
    """Drop trailing tokens that are ordinary words rather than name parts."""
    toks = name.split()
    while toks and toks[-1].lower() in _STOP:
        toks.pop()
    return " ".join(toks)


def detect_names(query: str) -> List[str]:
    """Person names in employee-supplied text, found via explicit triggers only."""
    found: List[str] = []
    if not isinstance(query, str):
        return found
    for m in _TRIGGERS.finditer(query):
        name = _trim(m.group("name"))
        if not name or name.lower() in _STOP or not name[0].isupper():
            continue
        if name not in found:
            found.append(name)
    # Longest first, so "Michael Chen" is replaced before a bare "Michael".
    return sorted(found, key=len, reverse=True)


def redact_text(text: str, names: List[str] = ()) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat, tag in _STRUCTURED:
        out = pat.sub(tag, out)
    for n in names:
        out = re.sub(r"\b" + re.escape(n) + r"\b", "[EMPLOYEE_NAME]", out)
    return out


def redact_obj(obj: Any, names: List[str] = ()) -> Any:
    if isinstance(obj, str):
        return redact_text(obj, names)
    if isinstance(obj, dict):
        return {k: redact_obj(v, names) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, names) for v in obj]
    return obj


def redact_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Redact a whole trace record. Returns (redacted_record, audit)."""
    names = detect_names(record.get("query", ""))
    redacted = redact_obj(record, names)
    audit = {
        "redaction_version": REDACTION_VERSION,
        "names_detected": len(names),
        "applied_before_write": True,
    }
    return redacted, audit
