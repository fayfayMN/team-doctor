"""RACI health checks — deterministic and explainable, no LLM.

A RACI matrix is only useful if someone checks it for the failure patterns that
quietly break teams. These rules encode the well-known ones so the app can say
*exactly* what's wrong and why.

The matrix is stored as ``raci[workstream_id][member_id] = {codes}`` where each
cell is a SET of RACI_CODES keys ("A","R","C","I"). A set (not a single code)
because one person can be both Accountable and Responsible on the same workstream.
"""

from __future__ import annotations

from typing import Dict, List

# A member who is Accountable for more than this many workstreams is a bottleneck.
OVERLOAD_THRESHOLD = 3


def _assignments_for(raci: Dict, workstream_id: str) -> Dict[str, set]:
    """Return {member_id: set_of_codes} for a workstream, dropping empty cells.

    Tolerates a legacy single-code string per cell by wrapping it in a set."""
    out: Dict[str, set] = {}
    for m, codes in raci.get(workstream_id, {}).items():
        s = {codes} if isinstance(codes, str) else set(codes or ())
        s.discard("")
        if s:
            out[m] = s
    return out


def check(raci: Dict, workstreams: List, members: List) -> Dict:
    """Return {findings: [...], score: 0-1, summary: str}.

    findings = [{level: "error"|"warn"|"ok", msg: str}]
    Errors are structural breaks; warnings are risks worth a conversation.
    """
    findings: List[Dict] = []
    name = {m.id: m.name for m in members}

    if not workstreams:
        return {"findings": [{"level": "warn", "msg": "No workstreams yet — add the "
                              "areas of work this team owns."}], "score": 0.0,
                "summary": "Nothing to check."}

    # ── Per-workstream structural rules ───────────────────────────────────────
    for w in workstreams:
        a = [name.get(m, m) for m, cs in _assignments_for(raci, w.id).items() if "A" in cs]
        r = [name.get(m, m) for m, cs in _assignments_for(raci, w.id).items() if "R" in cs]

        if len(a) == 0:
            findings.append({"level": "error",
                             "msg": f"**{w.name}** has no *Accountable* owner — "
                                    "no single person answers for the outcome."})
        elif len(a) > 1:
            findings.append({"level": "error",
                             "msg": f"**{w.name}** has {len(a)} *Accountable* "
                                    f"owners ({', '.join(a)}). Accountability splits "
                                    "into finger-pointing — pick one."})

        if len(r) == 0:
            findings.append({"level": "warn",
                             "msg": f"**{w.name}** has no one *Responsible* — who "
                                    "actually does the work?"})

    # ── Cross-workstream load rules ───────────────────────────────────────────
    a_count: Dict[str, int] = {}
    involved: set = set()
    for w in workstreams:
        for m, cs in _assignments_for(raci, w.id).items():
            involved.add(m)
            if "A" in cs:
                a_count[m] = a_count.get(m, 0) + 1

    for m, n in a_count.items():
        if n > OVERLOAD_THRESHOLD:
            findings.append({"level": "warn",
                             "msg": f"**{name.get(m, m)}** is Accountable for {n} "
                                    "workstreams — single point of failure / hero "
                                    "risk. Spread ownership."})

    for m in members:
        if m.id not in involved:
            findings.append({"level": "warn",
                             "msg": f"**{m.name}** has no role on any workstream — "
                                    "uninvolved members drift toward free-riding."})

    # ── Score: structural completeness × ownership distribution ────────────
    # Per-workstream: each area needs exactly 1 Accountable + at least 1 Responsible.
    clean = 0
    for w in workstreams:
        a = [cs for cs in _assignments_for(raci, w.id).values() if "A" in cs]
        r = [cs for cs in _assignments_for(raci, w.id).values() if "R" in cs]
        if len(a) == 1 and len(r) >= 1:
            clean += 1
    structural = clean / len(workstreams)

    # Distribution: when one person owns everything across multiple workstreams,
    # it's a solo operation, not a team. Score it accordingly.
    distinct_a = len([m for m, n in a_count.items() if n > 0])
    if distinct_a <= 1 and len(workstreams) > 1:
        score = 0.0
        findings.append({"level": "error",
                         "msg": "One person is Accountable for every workstream — "
                                "this is a solo operation, not a team. Spread "
                                "ownership across at least two people before the "
                                "structure can be scored."})
    else:
        score = round(structural, 2)

    errors = sum(1 for f in findings if f["level"] == "error")
    if errors:
        summary = f"{errors} structural problem(s) to fix before you rely on this."
    elif findings:
        summary = "Structure is sound; some risks worth a conversation."
    else:
        findings.append({"level": "ok", "msg": "Clean RACI — every workstream has one "
                         "owner and at least one doer, and load is balanced."})
        summary = "Healthy."

    return {"findings": findings, "score": score, "summary": summary}
