"""Team health — the coach.

The coach looks at the team's *structure* (charter? RACI? decisions logged?) and
its *pulse*, and recommends **one** next practice, drawn from EOS (Entrepreneurial
Operating System) — the lightweight, small-business-native model — in order of
maturity. Coach-first, not feature-dump: one move at a time.

No LLM: every recommendation is a transparent rule with a stated "why".
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

# Suppress all results below this many responses, so no single response can be
# singled out. The core anonymity guarantee (used when a real pulse exists).
MIN_RESPONSES = 3

# 1..5 Likert. A dimension averaging below this is treated as a problem.
LOW = 3.0
SERIOUS = 2.5

QUESTIONS: List[Dict] = [
    {"id": "safety",        "dimension": "Psychological safety",
     "text": "I can raise problems, mistakes, or tough issues without fear of blame."},
    {"id": "role_clarity",  "dimension": "Role clarity",
     "text": "I know what I'm accountable for, and what everyone else owns."},
    {"id": "decisions",     "dimension": "Decision speed",
     "text": "We make decisions and move on, instead of stalling or revisiting endlessly."},
    {"id": "workload",      "dimension": "Workload fairness",
     "text": "Work is fairly shared — no one is silently carrying the team or coasting."},
    {"id": "direction",     "dimension": "Direction",
     "text": "I know our top priorities right now and how my work connects to them."},
    {"id": "communication", "dimension": "Communication",
     "text": "Information I need reaches me in time, in a place I can find it."},
    {"id": "credit",        "dimension": "Credit",
     "text": "Contributions are recognized fairly — credit follows the work."},
    {"id": "openness",      "dimension": "Leadership openness",
     "text": "When I give honest feedback, something actually changes."},
]

DIMENSION_OF = {q["id"]: q["dimension"] for q in QUESTIONS}


def aggregate(responses: List) -> Optional[Dict]:
    """Return per-dimension averages, or None if below the anonymity threshold."""
    if len(responses) < MIN_RESPONSES:
        return None
    dims: Dict[str, List[int]] = {q["id"]: [] for q in QUESTIONS}
    for r in responses:
        for qid, val in r.scores.items():
            if qid in dims:
                dims[qid].append(int(val))
    per_dim = {qid: round(mean(vals), 2) for qid, vals in dims.items() if vals}
    overall = round(mean(per_dim.values()), 2) if per_dim else 0.0
    return {"per_dimension": per_dim, "overall": overall, "n": len(responses)}


def _rec(title: str, why: str, practice: str, severity: str = "warn") -> Dict:
    return {"title": title, "why": why, "practice": practice, "severity": severity}


def coach(state: Dict) -> Dict:
    """Recommend the single next practice.

    Foundational gaps come first — there's no point measuring health before the
    basics (ownership, a record of decisions) exist.
    """
    pulse = state.get("pulse")
    also: List[str] = []

    if not state.get("has_charter"):
        return {"primary": _rec(
            "Write your charter",
            "There's no shared agreement on mission, values, and how you work. "
            "Everything else drifts without it.",
            "Spend 30 minutes writing four things where everyone can see them: "
            "(1) your mission in one sentence, (2) three to five values or ground "
            "rules, (3) how you'll decide when you disagree, and (4) how credit "
            "gets shared. Keep it to one page, and have every person say yes to it. "
            "(EOS — a lightweight operating system for small teams — calls this your "
            "'Vision': the shared agreement everything else builds on.)",
            "error"), "also": also, "maturity": "Forming"}

    if not state.get("has_workstreams") or state.get("raci_errors", 0) > 0:
        return {"primary": _rec(
            "Fix ownership (RACI)",
            "Some areas have no single owner, or ownership is split — the #1 source "
            "of free-riders and blame.",
            "Make a one-page ownership chart and give every area exactly one owner. "
            "Work the red flags above, in order: (1) for any area with no owner, "
            "assign one person now — even temporarily; (2) where two people share an "
            "area, pick the single person who makes the final call; (3) if someone "
            "owns too much, hand one of their areas to a member who has no role yet. "
            "(This is the EOS 'Accountability Chart' — one clear owner per function. "
            "EOS is a simple operating system for small teams.)",
            "error"), "also": also, "maturity": "Forming"}

    if state.get("decisions_count", 0) == 0:
        return {"primary": _rec(
            "Start a weekly issues + decisions rhythm",
            "Decisions aren't being recorded, so 'we never agreed to that' arguments "
            "and quiet revisionism are inevitable.",
            "Hold a short weekly meeting where you work through a list of issues and "
            "actually decide each one — then write the decision down so no one "
            "re-litigates it later. The recipe (EOS calls it a 'Level 10' meeting run "
            "with IDS): Identify the real issue, Discuss it briefly, then Solve it "
            "with a clear decision and one owner. Same day and time every week.",
            "warn"), "also": also, "maturity": "Operating"}

    if not state.get("has_rocks"):
        also.append("Add 3–7 quarterly Rocks so priorities are binary and owned.")
    if not state.get("has_scorecard"):
        also.append("Set up 5–15 scorecard numbers so the team sees reality weekly.")

    if not state.get("has_issues_resolved"):
        return {"primary": _rec(
            "Run your first Level 10 meeting",
            "You have structure and decisions on record, but no weekly issue-resolution "
            "rhythm yet. Without it, problems pile up instead of getting solved.",
            "Use a 60-minute weekly agenda. The core is IDS: raise an issue, discuss "
            "it with a timer, solve it with a decision + owner. Same day and time "
            "every week.",
            "warn"), "also": also, "maturity": "Operating"}

    if not pulse:
        return {"primary": _rec(
            "Run an anonymous pulse",
            f"You have the structure in place but no honest read on how the team "
            f"feels. Collect at least {MIN_RESPONSES} responses to unlock the signal.",
            "Share a short anonymous health pulse with the team. Results stay hidden "
            "until enough people respond.",
            "warn"), "also": also, "maturity": "Operating"}

    dim = pulse["per_dimension"]

    for qid, avg in sorted(dim.items(), key=lambda kv: kv[1]):
        if avg < LOW:
            also.append(f"{DIMENSION_OF[qid]} is low ({avg}/5)")

    if dim.get("openness", 5) < SERIOUS:
        return {"primary": _rec(
            "The bottleneck looks like leadership openness — not process",
            f"'When I give feedback, something changes' scored {dim['openness']}/5. "
            "Adding more process won't help if feedback goes nowhere; it may make "
            "things worse by looking like surveillance.",
            "For the leader: pick the lowest-scoring dimension and make ONE visible "
            "change the team asked for, then say so. If nothing changes over repeated "
            "cycles, that's information about the team, not a tooling problem.",
            "error"), "also": also, "maturity": "At risk"}

    low = [(qid, avg) for qid, avg in dim.items() if avg < LOW]
    if low:
        qid, avg = min(low, key=lambda kv: kv[1])
        practice = {
            "safety": "Leader goes first: openly own a mistake at the next meeting, "
                      "and thank people who raise problems. Safety is modeled, not declared.",
            "role_clarity": "Revisit the RACI together; make sure every member can "
                            "name what they're Accountable for.",
            "decisions": "Adopt the EOS Level 10 + IDS weekly rhythm so issues get "
                         "resolved instead of looping.",
            "workload": "Rebalance anyone Accountable for too much, and involve anyone "
                        "with no role.",
            "direction": "Adopt an EOS-style Scorecard (5–15 numbers everyone sees) "
                         "plus 3–7 quarterly Rocks so priorities are unmistakable.",
            "communication": "Write a short communication charter: which channel for "
                             "what, and expected response times. Default to async.",
            "credit": "Make credit explicit in the retro — each person presents the "
                      "part they led; record it.",
        }.get(qid, "Discuss this dimension openly at the next check-in.")
        return {"primary": _rec(
            f"Improve {DIMENSION_OF[qid].lower()}",
            f"It's your lowest-scoring dimension ({avg}/5).",
            practice, "warn"),
            "also": [a for a in also if not a.startswith(DIMENSION_OF[qid])],
            "maturity": "Operating"}

    return {"primary": _rec(
        "Healthy — keep the cadence",
        f"All dimensions are at or above neutral (overall {pulse['overall']}/5).",
        "Maintain the weekly rhythm and re-pulse each cycle.",
        "ok"), "also": also, "maturity": "Healthy"}
