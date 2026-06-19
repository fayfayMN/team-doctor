"""RACI health checks — deterministic and explainable, no LLM.

A RACI matrix is only useful if someone checks it for the failure patterns that
quietly break teams. These rules encode the well-known ones so the app can say
*exactly* what's wrong and why.

The matrix is stored as ``raci[workstream_id][member_id] = code`` where code is
one of the RACI_CODES keys ("A","R","C","I","").
"""

from __future__ import annotations

from typing import Dict, List

# A member who is Accountable for more than this many workstreams is a bottleneck.
OVERLOAD_THRESHOLD = 3


def _assignments_for(raci: Dict, workstream_id: str) -> Dict[str, str]:
    return {m: c for m, c in raci.get(workstream_id, {}).items() if c}


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
        a = [name.get(m, m) for m, c in _assignments_for(raci, w.id).items() if c == "A"]
        r = [name.get(m, m) for m, c in _assignments_for(raci, w.id).items() if c == "R"]

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
        for m, c in _assignments_for(raci, w.id).items():
            involved.add(m)
            if c == "A":
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

    # ── Score: fraction of workstreams with a clean ownership structure ───────
    clean = 0
    for w in workstreams:
        a = [c for c in _assignments_for(raci, w.id).values() if c == "A"]
        r = [c for c in _assignments_for(raci, w.id).values() if c == "R"]
        if len(a) == 1 and len(r) >= 1:
            clean += 1
    score = round(clean / len(workstreams), 2)

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
