"""Team health — the coach.

The coach looks at the team's *structure* (charter? RACI? decisions logged?) and
its *pulse*, and recommends **one** next practice, drawn from EOS (Entrepreneurial
Operating System) — the lightweight, small-business-native model — in order of
maturity. Coach-first, not feature-dump: one move at a time.

No LLM: every recommendation is a transparent rule with a stated "why".
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List, Optional

# Suppress all results below this many responses, so no single response can be
# singled out. The core anonymity guarantee (used when a real pulse exists).
MIN_RESPONSES = 3

# 1..5 Likert. A dimension averaging below this is treated as a problem.
LOW = 3.0
SERIOUS = 2.5

# At or below this many people, the full EOS apparatus (quarterly Rocks, a
# 5–15 metric Scorecard, a formal 60-minute Level 10) is oversized. A tiny team
# needs a decision log and a light check-in, not OKRs.
SMALL_TEAM = 4

# Words that signal the team is in crisis / transition, not steady-state. When
# present, stabilizing comes before any governance roadmap.
CRISIS_WORDS = (
    "resign", "stepped down", "step down", "quit", "walked away", "leaving",
    "leaderless", "collapse", "imploded", "dissolve", "falling apart",
    "fell apart", "disband", "no president", "without a president",
)


def continuity(text: str) -> Optional[Dict]:
    """If the situation describes a recent departure/collapse, return an
    emergency continuity checklist that must come BEFORE any governance work.
    Deterministic: a keyword trigger over the description, not AI judgment."""
    t = (text or "").lower()
    if not any(w in t for w in CRISIS_WORDS):
        return None
    return {
        "title": "Stabilize first — before any roadmap",
        "why": "Someone in a key role has left or the team is in transition. A few "
               "things break in the first days if no one handles them — do these "
               "before designing governance.",
        "steps": [
            "Declare an interim structure today — even “co-leads until we regroup.” "
            "A named stand-in stops the team from stalling.",
            "Audit account access: who controls email, Discord/Slack, LinkedIn, "
            "Notion, the website, shared drives, and any registration or payment "
            "logins? Make sure at least two remaining people can get into each.",
            "Notify your faculty advisor or sponsor in writing — short and factual.",
            "Check eligibility rules: many student orgs need a minimum number of "
            "officers (often four) and current registration to keep funding and room "
            "booking. Confirm you still qualify and fix gaps before the deadline.",
            "Pause external posting until the remaining team agrees what goes out "
            "publicly and who approves it.",
            "Write down what happened and every decision you make this week, so the "
            "next person inherits a record, not a mystery.",
        ],
    }

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
    small = state.get("team_size", 99) <= SMALL_TEAM

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
        if small:
            return {"primary": _rec(
                "Set up a decision log + a light weekly check-in",
                "Decisions aren't being recorded, so 'we never agreed to that' "
                "arguments are inevitable — and at your size, a heavy meeting system "
                "would be overkill.",
                "Keep it light, right-sized for a small team: (1) start one shared "
                "“decisions & status” doc — every time you decide something, write one "
                "line: what, who, when. (2) Agree the rule that's probably missing: "
                "what needs sign-off from everyone vs. what either of you can just do. "
                "(3) Do a 15-minute check-in on a set day each week, live or async. "
                "That's the whole system at your size — skip quarterly planning and "
                "metrics dashboards until the team is bigger.",
                "warn"), "also": also, "maturity": "Operating"}
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

    # Rocks and a Scorecard are for teams big enough to need them — don't push
    # OKR machinery onto a two-person club.
    if not small:
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


# ── The full roadmap — for visitors who want to go deeper than the one next step ─
# A sequenced EOS-style path. The coach above always names the single most
# important move; this lays out the whole ladder so a motivated team can see
# what comes after it. Each stage carries detailed, plain-language steps. Still
# 100% deterministic — every stage's "done?" is a transparent rule over state.
ROADMAP: List[Dict] = [
    {"key": "charter", "title": "1. Write a one-page charter (EOS calls this the Vision)",
     "done": lambda s: bool(s.get("has_charter")),
     "what": "A single page everyone agrees on: why the team exists and how it works.",
     "why": "Without a shared agreement, every later decision quietly drifts and "
            "people fill the gaps with different assumptions.",
     "steps": [
         "Mission in one sentence — what this team is here to do.",
         "Three to five values or ground rules you'll actually hold each other to.",
         "Your decision rule: when you disagree, how does a decision get made?",
         "Your credit rule: how does recognition follow the work?",
         "Put it where everyone can see it, and have each person say yes out loud."]},
    {"key": "raci", "title": "2. Give every area one clear owner (EOS Accountability Chart)",
     "done": lambda s: bool(s.get("has_workstreams")) and s.get("raci_errors", 0) == 0,
     "what": "A one-page map: each area of work has exactly one Accountable owner "
             "and at least one person doing it.",
     "why": "Shared or missing ownership is the #1 source of free-riders, dropped "
            "balls, and blame. One name per area ends 'I thought you had it.'",
     "steps": [
         "List every area of work the team owns.",
         "For each, name ONE Accountable owner — the person who answers for it.",
         "Name who's Responsible (does the hands-on work); it can be the same person.",
         "Fix the red flags first: unowned areas, two owners, or one person owning "
         "too much.",
         "Re-run this check until the structure score is 100%."]},
    {"key": "weekly", "title": "3. Run a 60-minute weekly meeting (EOS Level 10 + IDS)",
     "done": lambda s: s.get("decisions_count", 0) > 0 or s.get("has_issues_resolved"),
     "what": "Same day, same time each week. You work a list of issues and decide "
             "each one — then write the decision down.",
     "why": "Issues that don't have a standing time to get solved pile up, and "
            "undocumented decisions get silently re-litigated.",
     "steps": [
         "Keep a running list of issues anyone can add to during the week.",
         "Each week, pick the most important issues and run IDS on them: "
         "Identify the real issue, Discuss it briefly, then Solve it.",
         "Every solve ends with a clear decision AND one owner.",
         "Write the decision in a shared log so no one re-opens it later.",
         "Protect the time — same slot every week, no skipping."]},
    {"key": "rocks", "title": "4. Set 3–7 quarterly priorities (EOS Rocks)",
     "done": lambda s: bool(s.get("has_rocks")),
     "what": "A short list of the few things that must get done this quarter — "
             "each with one owner and a yes/no 'done' definition.",
     "why": "Without a few named priorities, everything feels urgent and the "
            "important-but-not-loud work never happens.",
     "steps": [
         "As a team, pick the 3–7 outcomes that matter most for the next 90 days.",
         "Give each Rock a single owner and a clear 'done looks like…' line.",
         "Review Rock status at the top of the weekly meeting (on track / off track).",
         "Anything off-track becomes an issue to IDS that week.",
         "At quarter end, score them honestly and set the next set."]},
    {"key": "scorecard", "title": "5. Track 5–15 weekly numbers (EOS Scorecard)",
     "done": lambda s: bool(s.get("has_scorecard")),
     "what": "A handful of numbers, each owned by one person, reviewed weekly — "
             "so you see reality before it becomes a crisis.",
     "why": "Opinions argue; numbers settle. A weekly pulse of real metrics catches "
            "problems while they're still small.",
     "steps": [
         "Pick 5–15 numbers that tell you the team is healthy (leads, output, "
         "cash, response time — whatever matters here).",
         "Give each number one owner who reports it weekly.",
         "Set an expected range for each so 'off' is obvious at a glance.",
         "Glance at the scorecard at the start of every weekly meeting.",
         "Any number out of range becomes an issue to solve."]},
    {"key": "pulse", "title": "6. Take an honest read on how the team feels (anonymous pulse)",
     "done": lambda s: bool(s.get("pulse")),
     "what": "A short, anonymous survey on safety, clarity, workload, and trust — "
             "results stay hidden until enough people respond.",
     "why": "Structure can look perfect on paper while people quietly burn out or "
            "stop speaking up. The pulse surfaces what the org chart can't.",
     "steps": [
         "Send a short anonymous health check to everyone on the team.",
         "Wait for at least three responses so no one can be singled out.",
         "Look at the lowest-scoring dimension first — that's your next conversation.",
         "Make ONE visible change people asked for, and say you did it.",
         "Re-pulse each cycle to see whether it actually moved."]},
]


# For a tiny team, stage 3 is a light check-in (not a formal Level 10) and the
# Rocks/Scorecard stages are dropped entirely — they're machinery a 2–4 person
# team doesn't need yet.
LIGHT_WEEKLY = {
    "key": "weekly", "title": "3. Keep a decision log + a short weekly check-in",
    "done": lambda s: s.get("decisions_count", 0) > 0 or s.get("has_issues_resolved"),
    "what": "One shared doc for decisions and status, plus a 15-minute check-in on a "
            "set day — live or async.",
    "why": "At your size this is the entire operating system you need. It stops "
           "'we never agreed to that' without the overhead of a formal meeting.",
    "steps": [
        "Start one “decisions & status” doc everyone can see and edit.",
        "Every decision gets one line: what was decided, who owns it, when.",
        "Agree the rule that's usually missing: what needs everyone's sign-off vs. "
        "what either person can just do.",
        "Do a 15-minute check-in once a week on a set day.",
        "Skip quarterly planning and metric dashboards until the team grows."]}

# Replaces Rocks + Scorecard for a small team: a couple of semester goals and a
# few simple health signals — not OKRs and a metrics dashboard.
LIGHT_DIRECTION = {
    "key": "direction", "title": "4. Set 2–3 semester goals + watch 2–3 health signals",
    "done": lambda s: bool(s.get("has_rocks")) or bool(s.get("has_scorecard")),
    "what": "A couple of goals for the whole semester, plus a few simple signals you "
            "glance at each session — sized for a small club, not a company.",
    "why": "You need direction and early warning, not quarterly OKRs or a 15-metric "
           "dashboard that no one at your size will keep up.",
    "steps": [
        "Pick 2–3 goals for the SEMESTER (e.g. “run 4 sessions”, “grow to 20 active "
        "members”, “send 2 teams to a competition”) — not quarterly Rocks.",
        "Pick 2–3 health signals to watch: attendance per session, officer response "
        "time, and account-access coverage (who can get into each tool).",
        "Glance at them at each check-in; if one slips, it becomes the next decision."]}


def roadmap(state: Dict) -> List[Dict]:
    """Return the path with each stage marked done / 'now' / 'next'.

    Deterministic: 'done' is a rule over state; the first not-done stage is where
    the team is now. Right-sizes for small teams — a tiny club sees a light
    decision-log step and skips the Rocks/Scorecard machinery.
    """
    small = state.get("team_size", 99) <= SMALL_TEAM
    stages = []
    for stage in ROADMAP:
        if small and stage["key"] in ("rocks", "scorecard"):
            continue
        stages.append(LIGHT_WEEKLY if (small and stage["key"] == "weekly") else stage)
    if small:
        # One light "direction" stage stands in for Rocks + Scorecard, placed before
        # the pulse stage.
        idx = next((i for i, s in enumerate(stages) if s["key"] == "pulse"), len(stages))
        stages.insert(idx, LIGHT_DIRECTION)

    out: List[Dict] = []
    first_open = True
    for n, stage in enumerate(stages, start=1):
        is_done = bool(stage["done"](state))
        if is_done:
            status = "done"
        elif first_open:
            status, first_open = "now", False
        else:
            status = "next"
        # Renumber the title so dropping stages doesn't leave gaps (1, 2, 3…).
        title = re.sub(r"^\s*\d+\.\s*", f"{n}. ", stage["title"])
        out.append({
            "key": stage["key"], "title": title, "status": status,
            "what": stage["what"], "why": stage["why"], "steps": stage["steps"]})
    return out
