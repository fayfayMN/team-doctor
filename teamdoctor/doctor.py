"""Team Doctor — turn a plain-English team description into a real diagnosis.

The LLM does only two language jobs:
  1. structure — messy description -> {members, workstreams, raci}
  2. narration — answer follow-ups using the engine's findings as ground truth

The diagnosis itself — every flag, every recommendation — comes from the
deterministic engine (raci.check, health.coach). The model never invents a
finding, so the "explainable, can't hallucinate" promise holds.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List

from teamdoctor import health, llm
from teamdoctor import raci as raci_check
from teamdoctor.models import Member, Workstream

# Real-world-style case studies — vivid, messy, the way someone would actually
# describe their team out loud. Each lights up different findings across layers.
SAMPLES: dict = {
    "Series A startup (people burning out)": (
        "We're an 8-person B2B software startup that just raised a seed round. Our "
        "CEO Priya is great with investors but also jumps into every sales call, so "
        "both she and our head of sales Sven think they own the sales number — deals "
        "stall when they disagree. Our CTO Tom is brilliant but owns the platform, "
        "the infrastructure, the product roadmap, AND all hiring — he's the "
        "bottleneck for everything and he's exhausted. Two engineers, Jade and "
        "Marco, do great work but only when Tom tells them exactly what to do. Lena "
        "is our product manager but has no real authority — Tom overrides her. Ivy "
        "runs marketing completely alone with no clear goals. Ben handles customer "
        "success and is drowning in tickets. Nobody owns finance — we almost missed "
        "payroll last month. We never write decisions down, so we keep relitigating "
        "the same arguments, and people are starting to burn out."
    ),
    "Nonprofit board (co-chairs, slow decisions)": (
        "We're the volunteer board of a community arts nonprofit, about six people. "
        "Our two co-chairs, Dana and Paul, jointly run everything — every decision "
        "needs both of them to agree, so things move slowly and sometimes nothing "
        "happens at all. Our treasurer Ana is barely involved; she gets handed the "
        "books once a year. Our volunteer coordinator Kim does an amazing job but is "
        "completely overloaded — she runs volunteers, events, AND community outreach "
        "by herself. Nobody really owns grant writing, even though that's where most "
        "of our funding should come from. We don't track anything week to week, and "
        "we only meet when there's a crisis."
    ),
    "University club (volunteers, churn)": (
        "We're a university robotics club — around seven active members, but it "
        "changes every semester. Our president Alex basically does everything: "
        "recruiting, sponsorships, build planning, and running every meeting — and "
        "gets burnt out every single year. The other officers have titles like 'VP' "
        "and 'Treasurer' but nobody's really sure what they actually do. We have two "
        "or three members who show up but never get assigned anything, so they drift "
        "away. We argue about design decisions for weeks and never actually decide. "
        "When seniors graduate, all their knowledge leaves with them because nothing "
        "is written down."
    ),
}

# Default sample (first one) kept for backward compatibility.
SAMPLE_TEAM = next(iter(SAMPLES.values()))

# Pre-structured sample teams that run through the DETERMINISTIC engine with NO
# AI call and NO API key — used for the free, key-less paths (samples + booth).
SAMPLE_SPECS: dict = {
    "Series A startup (people burning out)": {
        "team_name": "Vectorly", "mission": "B2B analytics that ships weekly",
        "summary": "CTO overloaded, sales ownership split, nobody owns finance",
        "members": [{"name": "Priya", "role": "CEO"}, {"name": "Tom", "role": "CTO"},
                    {"name": "Jade", "role": "Eng"}, {"name": "Marco", "role": "Eng"},
                    {"name": "Lena", "role": "Product"}, {"name": "Sven", "role": "Sales"},
                    {"name": "Ivy", "role": "Marketing"}, {"name": "Ben", "role": "CS"}],
        "workstreams": [{"name": "Platform"}, {"name": "Infra"}, {"name": "Product"},
                        {"name": "Hiring"}, {"name": "Sales"}, {"name": "Marketing"},
                        {"name": "Customer success"}, {"name": "Finance"}],
        "raci": [
            {"workstream": "Platform", "member": "Tom", "code": "A"},
            {"workstream": "Platform", "member": "Jade", "code": "R"},
            {"workstream": "Infra", "member": "Tom", "code": "A"},
            {"workstream": "Infra", "member": "Marco", "code": "R"},
            {"workstream": "Product", "member": "Tom", "code": "A"},
            {"workstream": "Product", "member": "Lena", "code": "R"},
            {"workstream": "Hiring", "member": "Tom", "code": "A"},
            {"workstream": "Sales", "member": "Sven", "code": "A"},
            {"workstream": "Sales", "member": "Priya", "code": "A"},
            {"workstream": "Marketing", "member": "Ivy", "code": "A"},
            {"workstream": "Customer success", "member": "Ben", "code": "A"}],
        "charter": {
            "mission": "Help analysts get answers in seconds.",
            "values": ["Ship weekly", "Own your area", "Customer truth"],
            "decision_rule": "The accountable owner decides after a 10-minute debate.",
            "communication_rule": "Async by default; one weekly sync.",
            "credit_rule": "Each person presents the part they led."},
        "issues": [
            {"issue": "Tom is a single point of failure", "suggested_owner": "Tom",
             "next_step": "Hand infra to Marco and hiring to Priya with a checklist"},
            {"issue": "Sales ownership is split between Priya and Sven", "suggested_owner": "Priya",
             "next_step": "Make Sven solely accountable for the sales number"},
            {"issue": "Nobody owns finance — almost missed payroll", "suggested_owner": "Ben",
             "next_step": "Assign finance ownership this week"}],
    },
    "Nonprofit board (co-chairs, slow decisions)": {
        "team_name": "Riverside Arts", "mission": "Fund community art programs",
        "summary": "Two co-chairs jointly own everything; treasurer sidelined",
        "members": [{"name": "Dana", "role": "Co-chair"}, {"name": "Paul", "role": "Co-chair"},
                    {"name": "Ana", "role": "Treasurer"}, {"name": "Kim", "role": "Volunteer lead"}],
        "workstreams": [{"name": "Fundraising"}, {"name": "Events"},
                        {"name": "Grants"}, {"name": "Volunteers"}],
        "raci": [
            {"workstream": "Fundraising", "member": "Dana", "code": "A"},
            {"workstream": "Fundraising", "member": "Paul", "code": "A"},
            {"workstream": "Events", "member": "Dana", "code": "A"},
            {"workstream": "Events", "member": "Paul", "code": "A"},
            {"workstream": "Grants", "member": "Dana", "code": "A"},
            {"workstream": "Grants", "member": "Paul", "code": "A"},
            {"workstream": "Volunteers", "member": "Kim", "code": "A"},
            {"workstream": "Volunteers", "member": "Kim", "code": "R"}],
        "charter": {
            "mission": "Bring more art to the community, sustainably.",
            "values": ["Transparency", "Shared load", "Decide and move"],
            "decision_rule": "One named owner per area decides; co-chairs break ties only.",
            "communication_rule": "Monthly board meeting + a shared decisions doc.",
            "credit_rule": "Recognize volunteers publicly at each event."},
        "issues": [
            {"issue": "Every decision needs both co-chairs, so things stall", "suggested_owner": "Dana",
             "next_step": "Split areas: give each co-chair sole ownership of specific functions"},
            {"issue": "Treasurer Ana is barely involved", "suggested_owner": "Ana",
             "next_step": "Give Ana ownership of a finance/grants workstream"},
            {"issue": "Kim is overloaded across volunteers, events, outreach", "suggested_owner": "Kim",
             "next_step": "Delegate outreach to a new volunteer owner"}],
    },
    "University club (volunteers, churn)": {
        "team_name": "Robotics Club", "mission": "Build competitive robots and grow members",
        "summary": "President does everything; officers undefined; members drift; knowledge lost",
        "members": [{"name": "Alex", "role": "President"}, {"name": "Sam", "role": "VP"},
                    {"name": "Riley", "role": "Treasurer"}, {"name": "Jordan", "role": "Member"},
                    {"name": "Casey", "role": "Member"}],
        "workstreams": [{"name": "Recruiting"}, {"name": "Sponsorships"},
                        {"name": "Build planning"}, {"name": "Meetings"}, {"name": "Knowledge"}],
        "raci": [
            {"workstream": "Recruiting", "member": "Alex", "code": "A"},
            {"workstream": "Sponsorships", "member": "Alex", "code": "A"},
            {"workstream": "Build planning", "member": "Alex", "code": "A"},
            {"workstream": "Meetings", "member": "Alex", "code": "A"}],
        "charter": {
            "mission": "Design and build competitive robots while teaching new members.",
            "values": ["Share responsibility", "Decide together", "Document everything"],
            "decision_rule": "Design choices: majority vote after a one-week limit; lead breaks ties.",
            "communication_rule": "Weekly meeting + a shared drive updated within 48 hours.",
            "credit_rule": "Everyone who works on a build is named on the project page."},
        "issues": [
            {"issue": "Alex owns everything and burns out yearly", "suggested_owner": "Alex",
             "next_step": "Delegate recruiting to Sam and sponsorships to Riley"},
            {"issue": "Officer roles are undefined", "suggested_owner": "Alex",
             "next_step": "Write a one-page role description for VP and Treasurer"},
            {"issue": "Knowledge is lost when seniors graduate", "suggested_owner": "Casey",
             "next_step": "Set up a shared drive with build/meeting templates"}],
    },
    "Well-run team (healthy — shows the next step)": {
        "team_name": "Bright Pod", "mission": "Run paid ads + content for 6 clients, profitably",
        "summary": "Roles are clear and balanced; the team wants to know what to tighten next",
        "members": [{"name": "Nora", "role": "Pod lead"}, {"name": "Sam", "role": "Ads"},
                    {"name": "Tia", "role": "Content"}, {"name": "Omar", "role": "Design"}],
        "workstreams": [{"name": "Paid ads"}, {"name": "Content"},
                        {"name": "Design"}, {"name": "Client reporting"}],
        "raci": [
            {"workstream": "Paid ads", "member": "Sam", "code": "A"},
            {"workstream": "Paid ads", "member": "Nora", "code": "R"},
            {"workstream": "Content", "member": "Tia", "code": "A"},
            {"workstream": "Content", "member": "Omar", "code": "R"},
            {"workstream": "Design", "member": "Omar", "code": "A"},
            {"workstream": "Design", "member": "Tia", "code": "R"},
            {"workstream": "Client reporting", "member": "Nora", "code": "A"},
            {"workstream": "Client reporting", "member": "Sam", "code": "R"}],
        "charter": {
            "mission": "Make six clients measurably more money, every month.",
            "values": ["Own your number", "Show the work", "No surprises"],
            "decision_rule": "The area owner decides; escalate only when it crosses two areas.",
            "communication_rule": "Async updates daily; one 30-minute weekly sync.",
            "credit_rule": "Each owner presents their client's results at the weekly sync."},
        "issues": [
            {"issue": "No weekly rhythm yet to catch problems early", "suggested_owner": "Nora",
             "next_step": "Start a 30-minute weekly review of each client's numbers"},
            {"issue": "Wins and decisions aren't written down", "suggested_owner": "Nora",
             "next_step": "Keep a one-line decision log after each sync"}],
    },
    "Small business (nobody owns the money)": {
        "team_name": "Maple Café", "mission": "A neighborhood café people keep coming back to",
        "summary": "Service runs well, but nobody owns the finances",
        "members": [{"name": "Rosa", "role": "Owner"}, {"name": "Dev", "role": "Manager"},
                    {"name": "Lin", "role": "Head cook"}, {"name": "Kai", "role": "Barista lead"}],
        "workstreams": [{"name": "Kitchen"}, {"name": "Front of house"}, {"name": "Marketing"},
                        {"name": "Ordering & supplies"}, {"name": "Finance"}],
        "raci": [
            {"workstream": "Kitchen", "member": "Lin", "code": "A"},
            {"workstream": "Kitchen", "member": "Kai", "code": "R"},
            {"workstream": "Front of house", "member": "Dev", "code": "A"},
            {"workstream": "Front of house", "member": "Kai", "code": "R"},
            {"workstream": "Marketing", "member": "Rosa", "code": "A"},
            {"workstream": "Marketing", "member": "Dev", "code": "R"},
            {"workstream": "Ordering & supplies", "member": "Dev", "code": "A"},
            {"workstream": "Ordering & supplies", "member": "Lin", "code": "R"}],
        "charter": {
            "mission": "Be the café this neighborhood can't imagine losing.",
            "values": ["Warm service", "Clean books", "No waste"],
            "decision_rule": "Rosa decides on money and brand; area leads decide day-to-day.",
            "communication_rule": "A 10-minute pre-shift huddle; a weekly numbers check.",
            "credit_rule": "Shout out the person behind a great week at the huddle."},
        "issues": [
            {"issue": "Nobody owns finance — cash and bills get handled late", "suggested_owner": "Rosa",
             "next_step": "Assign finance ownership and set a weekly money check"},
            {"issue": "Marketing is ad hoc with no plan", "suggested_owner": "Dev",
             "next_step": "Pick one channel and post on a set schedule"}],
    },
    "The overloaded star (looks fine, one big risk)": {
        "team_name": "Forge Studio", "mission": "Design and ship client web apps",
        "summary": "Everything has an owner — but it's nearly all one person",
        "members": [{"name": "Dana", "role": "Founder"}, {"name": "Eli", "role": "Engineer"},
                    {"name": "Mara", "role": "Designer"}, {"name": "Nick", "role": "PM"}],
        "workstreams": [{"name": "Strategy"}, {"name": "Design"}, {"name": "Build"},
                        {"name": "Delivery"}, {"name": "Client comms"}],
        "raci": [
            {"workstream": "Strategy", "member": "Dana", "code": "A"},
            {"workstream": "Strategy", "member": "Eli", "code": "R"},
            {"workstream": "Design", "member": "Dana", "code": "A"},
            {"workstream": "Design", "member": "Mara", "code": "R"},
            {"workstream": "Build", "member": "Dana", "code": "A"},
            {"workstream": "Build", "member": "Nick", "code": "R"},
            {"workstream": "Delivery", "member": "Dana", "code": "A"},
            {"workstream": "Delivery", "member": "Eli", "code": "R"},
            {"workstream": "Client comms", "member": "Mara", "code": "A"},
            {"workstream": "Client comms", "member": "Nick", "code": "R"}],
        "charter": {
            "mission": "Ship web apps clients love, without heroics.",
            "values": ["Share the load", "Document as you go", "Steady over crunch"],
            "decision_rule": "The area owner decides; Dana stops being the default approver.",
            "communication_rule": "A short daily check-in; a weekly planning session.",
            "credit_rule": "Each person owns and presents the part they led."},
        "issues": [
            {"issue": "Dana is Accountable for four of five areas — a single point of failure",
             "suggested_owner": "Dana", "next_step": "Hand Strategy to Eli and Build to Nick as owners"},
            {"issue": "If Dana is out, four workstreams stall", "suggested_owner": "Dana",
             "next_step": "Name a backup owner for each area Dana holds"}],
    },
}

EXTRACT_SYSTEM = """You are the intake step of a team-health tool. Turn the \
user's description of their team into structured data. Reply with ONE JSON \
object and nothing else.

Schema:
{
  "ready": boolean,           // true if you have enough to map the team
  "follow_up": string,        // if not ready, ONE short friendly question
  "team_name": string,
  "members": [{"name": string, "role": string}],
  "workstreams": [{"name": string, "description": string}],
  "raci": [{"workstream": string, "member": string, "code": "A"|"R"|"C"|"I"}],
  "mission": string,          // "" if none mentioned
  "summary": string           // one sentence on what they say is going wrong
}

Rules:
- RACI codes: A=Accountable (owns the outcome, ideally exactly one per \
workstream), R=Responsible (does the work), C=Consulted, I=Informed.
- In "raci", spell workstream and member names EXACTLY as you listed them above.
- Infer obvious ownership from the description (e.g. "Sara runs marketing" => \
Sara is A on a Marketing workstream; if someone clearly does the hands-on work, \
mark them R). Do NOT invent people or work they didn't mention.
- If the description is too vague to identify members or work, set ready=false \
and ask ONE concise follow_up question.
- Be faithful: only structure what they actually said."""

ANSWER_SYSTEM = """You are Team Doctor, a warm, plain-spoken team coach. You are \
given a deterministic diagnosis of the user's team (findings + a single \
recommended next step) produced by a rules engine. Answer using ONLY those \
findings as ground truth — never invent new problems or numbers. Be encouraging, \
concrete, and brief."""


def extract(provider: str, model: str, api_key: str,
            conversation: List[Dict]) -> dict:
    """Ask the model to turn the conversation so far into a team spec."""
    messages = [{"role": "system", "content": EXTRACT_SYSTEM}] + conversation
    raw = llm.chat(provider, model, api_key, messages,
                   temperature=0.2, json_mode=True)
    spec = llm.extract_json(raw)
    if spec is None:
        raise llm.LLMError("The model didn't return usable structure. Try "
                           "rephrasing, or switch to a stronger model.")
    return spec


def diagnose(spec: dict) -> dict:
    """Run the deterministic engine on an extracted spec."""
    junk = {"", "none", "unnamed", "untitled", "n/a", "na", "tbd", "null"}

    def _clean(v) -> str:
        return (str(v) if v is not None else "").strip()

    members = [Member.create(_clean(m.get("name")), _clean(m.get("role")))
               for m in spec.get("members", [])
               if _clean(m.get("name")).lower() not in junk]
    workstreams = [Workstream.create(_clean(w.get("name")), _clean(w.get("description")))
                   for w in spec.get("workstreams", [])
                   if _clean(w.get("name")).lower() not in junk]

    member_id = {m.name.lower(): m.id for m in members}
    ws_id = {w.name.lower(): w.id for w in workstreams}

    def _match(name, idmap):
        """Exact name match, then a tolerant contains-match so a slightly
        reworded reference (AI) still links to the right item."""
        key = _clean(name).lower()
        if not key:
            return None
        if key in idmap:
            return idmap[key]
        for nm, _id in idmap.items():
            if key in nm or nm in key:
                return _id
        return None

    # Each cell holds a SET of codes: the same person can be both Accountable
    # AND Responsible on one workstream (the natural case for a one-person task),
    # so a single code per cell would let "R" overwrite "A" and drop the owner.
    raci: Dict[str, Dict[str, set]] = {w.id: {} for w in workstreams}
    for row in spec.get("raci", []):
        wid = _match(row.get("workstream"), ws_id)
        mid = _match(row.get("member"), member_id)
        code = _clean(row.get("code")).upper()
        if wid and mid and code in ("A", "R", "C", "I"):
            raci.setdefault(wid, {}).setdefault(mid, set()).add(code)

    raci_result = raci_check.check(raci, workstreams, members)
    raci_errors = sum(1 for f in raci_result["findings"] if f["level"] == "error")

    state = {
        "has_charter": bool(spec.get("mission")),
        "has_workstreams": bool(workstreams),
        "raci_errors": raci_errors,
        "team_size": len(members),
        "decisions_count": 0,
        "responses_count": 0,
        "pulse": None,
        "has_rocks": False,
        "has_scorecard": False,
        "has_issues_resolved": False,
    }
    coach = health.coach(state)
    roadmap = health.roadmap(state)
    # Crisis check: scan everything the user told us for signs of a recent
    # departure/collapse; if found, stabilizing comes before the roadmap.
    crisis_text = " ".join([
        spec.get("summary", "") or "", spec.get("mission", "") or "",
        " ".join(i.get("issue", "") for i in (spec.get("issues") or [])),
    ])
    continuity = health.continuity(crisis_text)

    return {
        "members": members,
        "workstreams": workstreams,
        "raci": raci,
        "raci_result": raci_result,
        "coach": coach,
        "roadmap": roadmap,
        "continuity": continuity,
        "team_size": len(members),
        "team_name": (spec.get("team_name") or "Your team").strip(),
        "mission": spec.get("mission", ""),
        "summary": spec.get("summary", ""),
    }


MAX_VALUES = 5
MAX_ISSUES = 6


def normalize_charter(d: dict) -> dict:
    """Validate the charter the agent produced; None if essentially empty."""
    d = d or {}
    values = [str(v).strip() for v in d.get("values", []) if str(v).strip()][:MAX_VALUES]
    charter = {
        "mission": str(d.get("mission", "")).strip(),
        "values": values,
        "decision_rule": str(d.get("decision_rule", "")).strip(),
        "communication_rule": str(d.get("communication_rule", "")).strip(),
        "credit_rule": str(d.get("credit_rule", "")).strip(),
    }
    if not any([charter["mission"], values, charter["decision_rule"],
                charter["communication_rule"], charter["credit_rule"]]):
        return None
    return charter


def normalize_issues(lst: list) -> list:
    """Validate the issues list (IDS): each needs a title; owner defaults to TBD."""
    out: List[Dict] = []
    for it in (lst or [])[:MAX_ISSUES]:
        title = str(it.get("issue", "")).strip()
        if not title:
            continue
        out.append({
            "issue": title,
            "suggested_owner": str(it.get("suggested_owner", "TBD")).strip() or "TBD",
            "next_step": str(it.get("next_step", "")).strip(),
        })
    return out or None


def build_workspace(data: dict) -> tuple:
    """Assemble a full workspace from ONE agent response, running the deterministic
    engine locally. Returns (workspace, skills_applied)."""
    charter = normalize_charter(data.get("charter"))
    # If the agent only put the mission in the charter, feed it back so the coach
    # knows a foundation exists.
    if charter and charter.get("mission") and not data.get("mission"):
        data["mission"] = charter["mission"]

    diagnosis = diagnose(data)
    issues = normalize_issues(data.get("issues"))

    ws = {"spec": data, "diagnosis": diagnosis, "charter": charter, "issues": issues}

    skills = ["🧠 Read your team into structure"]
    if charter:
        skills.append("📜 Drafted a charter")
    if diagnosis:
        skills.append("🧩 Mapped ownership & ran the RACI check")
    if issues:
        skills.append("🔟 Surfaced issues (IDS)")
    if diagnosis and diagnosis["coach"].get("primary"):
        skills.append("🎯 Recommended the next practice")
    return ws, skills


def answer(provider: str, model: str, api_key: str, workspace: dict,
           conversation: List[Dict]) -> str:
    """Answer a follow-up question, grounded in everything the agent built."""
    context = workspace_context(workspace)
    messages = [{"role": "system",
                 "content": ANSWER_SYSTEM + "\n\nWHAT WE BUILT:\n" + context}]
    messages += conversation
    return llm.chat(provider, model, api_key, messages, temperature=0.4)


# ── text helpers ──────────────────────────────────────────────────────────────

def _strip_md(s: str) -> str:
    return re.sub(r"\*+", "", s)


def summary_text(diag: dict) -> str:
    r = diag["raci_result"]
    n_err = sum(1 for f in r["findings"] if f["level"] == "error")
    n_warn = sum(1 for f in r["findings"] if f["level"] == "warn")
    lines = [f"Here's what I see in **{diag['team_name']}**:",
             f"- RACI health: **{n_err}** structural error(s), **{n_warn}** risk(s)."]
    primary = diag["coach"].get("primary")
    if primary:
        lines.append(f"- **Start here:** {primary['title']}.")
    lines.append("Full breakdown is on the right → ask me anything below.")
    return "\n".join(lines)


def findings_context(diag: dict) -> str:
    r = diag["raci_result"]
    lines = [f"Team: {diag['team_name']}"]
    if diag.get("summary"):
        lines.append(f"Reported problem: {diag['summary']}")
    lines.append(f"RACI score: {r['score']} — {r['summary']}")
    for f in r["findings"]:
        lines.append(f"- [{f['level']}] {_strip_md(f['msg'])}")
    primary = diag["coach"].get("primary")
    if primary:
        lines.append(f"Top recommendation: {primary['title']} — "
                     f"{_strip_md(primary['why'])} Practice: {_strip_md(primary['practice'])}")
    for a in diag["coach"].get("also", []):
        lines.append(f"- also: {_strip_md(a)}")
    return "\n".join(lines)


def workspace_context(ws: dict) -> str:
    """Compact, ground-truth summary of everything the agent built — for Q&A."""
    lines: List[str] = []
    charter = ws.get("charter")
    if charter:
        lines.append("CHARTER:")
        if charter.get("mission"):
            lines.append(f"- Mission: {charter['mission']}")
        if charter.get("values"):
            lines.append(f"- Values: {', '.join(charter['values'])}")
        for key in ("decision_rule", "communication_rule", "credit_rule"):
            if charter.get(key):
                lines.append(f"- {key.replace('_', ' ').title()}: {charter[key]}")
    diag = ws.get("diagnosis")
    if diag:
        lines.append("")
        lines.append(findings_context(diag))
    issues = ws.get("issues")
    if issues:
        lines.append("")
        lines.append("ISSUES (IDS):")
        for it in issues:
            lines.append(f"- {it['issue']} (owner: {it['suggested_owner']}; "
                         f"next: {it['next_step']})")
    return "\n".join(lines) if lines else "Nothing built yet."


def report_md(ws: dict) -> str:
    """A take-home report a guest can download after the demo."""
    diag = ws.get("diagnosis") or {}
    team_name = diag.get("team_name") or "Your team"
    out = [f"# Team Doctor report — {team_name}",
           f"_Generated {date.today().isoformat()}_", ""]
    if diag.get("summary"):
        out += [f"**What you described:** {diag['summary']}", ""]

    charter = ws.get("charter")
    if charter:
        out += ["## Charter"]
        if charter.get("mission"):
            out += [f"**Mission:** {charter['mission']}", ""]
        if charter.get("values"):
            out += ["**Values:** " + ", ".join(charter["values"]), ""]
        for key in ("decision_rule", "communication_rule", "credit_rule"):
            if charter.get(key):
                out += [f"**{key.replace('_', ' ').title()}:** {charter[key]}"]
        out += [""]

    if diag:
        r = diag["raci_result"]
        out += [f"## RACI structure score: {round(r['score'] * 100)}%", ""]
        for f in r["findings"]:
            tag = {"error": "[FIX]", "warn": "[RISK]", "ok": "[OK]"}.get(f["level"], "-")
            out.append(f"- {tag} {_strip_md(f['msg'])}")
        out.append("")
        primary = diag["coach"].get("primary")
        if primary:
            out += ["## Start here",
                    f"**{primary['title']}**", "",
                    f"_Why:_ {_strip_md(primary['why'])}", "",
                    f"_Do this:_ {_strip_md(primary['practice'])}", ""]

    issues = ws.get("issues")
    if issues:
        out += ["## Issues to work (IDS)"]
        for it in issues:
            out += [f"- **{it['issue']}** — owner: {it['suggested_owner']}; "
                    f"next step: {it['next_step']}"]
        out += [""]

    out += ["---",
            "Built by Team Doctor. AI drafted the charter and issues; the RACI "
            "and coach findings are deterministic rules — traceable, not guessed.",
            "© 2026 Feifei Li."]
    return "\n".join(out)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _md_bold(s: str) -> str:
    """Escape, then turn **bold** markers into <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _esc(s))


def report_html(ws: dict) -> str:
    """A clean, self-contained HTML report — looks professional and prints to PDF."""
    diag = ws.get("diagnosis") or {}
    team = diag.get("team_name") or "Your team"
    charter = ws.get("charter")
    issues = ws.get("issues")

    badge = {"error": ("#A32D2D", "#FCEBEB", "FIX"),
             "warn": ("#854F0B", "#FAEEDA", "RISK"),
             "ok": ("#3B6D11", "#EAF3DE", "OK")}

    s = []
    s.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    s.append(f"<title>Team Doctor — {_esc(team)}</title>")
    s.append("<style>"
             "*{box-sizing:border-box}"
             "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
             "color:#1a1a1a;line-height:1.55;max-width:760px;margin:32px auto;padding:0 24px}"
             "h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 10px;"
             "border-bottom:2px solid #eee;padding-bottom:6px}"
             ".sub{color:#666;font-size:13px;margin:0 0 18px}"
             ".pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;"
             "border-radius:10px;margin-right:8px;vertical-align:middle}"
             ".finding{margin:6px 0;padding:8px 12px;border-radius:6px;background:#fafafa}"
             ".score{font-size:34px;font-weight:700;margin:4px 0}"
             ".chip{display:inline-block;background:#eef;border-radius:12px;padding:3px 10px;"
             "margin:3px 6px 3px 0;font-size:13px}"
             ".issue{margin:10px 0;padding:10px 14px;border-left:3px solid #534AB7;background:#f7f7fb}"
             ".issue .meta{color:#555;font-size:13px;margin-top:3px}"
             ".rule{margin:4px 0;font-size:14px}"
             "footer{margin-top:32px;border-top:1px solid #eee;padding-top:12px;"
             "color:#888;font-size:12px}"
             "@media print{body{margin:0}}"
             "</style></head><body>")

    s.append(f"<h1>🩺 Team Doctor — {_esc(team)}</h1>")
    s.append(f"<p class='sub'>Generated {date.today().isoformat()}</p>")
    if diag.get("summary"):
        s.append(f"<p><em>What you described:</em> {_esc(diag['summary'])}</p>")

    cont = diag.get("continuity")
    if cont:
        s.append("<div style='border:2px solid #A32D2D;background:#FCEBEB;"
                 "border-radius:8px;padding:12px 16px;margin:14px 0'>")
        s.append(f"<h2 style='border:none;margin:0 0 6px;color:#A32D2D'>🚨 "
                 f"{_esc(cont['title'])}</h2>")
        s.append(f"<p style='margin:2px 0'><strong>Why:</strong> {_esc(cont['why'])}</p>")
        s.append("<ul style='margin:6px 0'>"
                 + "".join(f"<li>{_esc(step)}</li>" for step in cont["steps"])
                 + "</ul></div>")

    if charter:
        s.append("<h2>📜 Charter</h2>")
        if charter.get("mission"):
            s.append(f"<p><strong>Mission:</strong> {_esc(charter['mission'])}</p>")
        if charter.get("values"):
            s.append("<p><strong>Values:</strong> "
                     + "".join(f"<span class='chip'>{_esc(v)}</span>"
                               for v in charter["values"]) + "</p>")
        for key, label in (("decision_rule", "Decisions"),
                           ("communication_rule", "Communication"),
                           ("credit_rule", "Credit")):
            if charter.get(key):
                s.append(f"<p class='rule'><strong>{label}:</strong> "
                         f"{_esc(charter[key])}</p>")

    if diag:
        r = diag["raci_result"]
        s.append("<h2>🧩 Ownership (RACI)</h2>")
        s.append(f"<div class='score'>{round(r['score'] * 100)}%"
                 "<span style='font-size:14px;color:#888'> structure score "
                 "(structural completeness only — risks flagged below)</span></div>")
        # The table behind the score — never show the number without the content.
        mname = {m.id: m.name for m in diag.get("members", [])}
        trows = []
        for w in diag.get("workstreams", []):
            cell = diag.get("raci", {}).get(w.id, {})
            a = ", ".join(mname.get(m, m) for m, cs in cell.items() if "A" in cs)
            rr = ", ".join(mname.get(m, m) for m, cs in cell.items() if "R" in cs)
            trows.append((w.name, a or "— none —", rr or "— none —"))
        if trows:
            s.append("<table style='border-collapse:collapse;width:100%;margin:8px 0;"
                     "font-size:14px'><thead><tr>"
                     "<th style='text-align:left;border-bottom:2px solid #ddd;padding:6px'>Area of work</th>"
                     "<th style='text-align:left;border-bottom:2px solid #ddd;padding:6px'>Accountable (owns it)</th>"
                     "<th style='text-align:left;border-bottom:2px solid #ddd;padding:6px'>Responsible (does it)</th>"
                     "</tr></thead><tbody>")
            for area, a, rr in trows:
                s.append(f"<tr><td style='border-bottom:1px solid #eee;padding:6px'>{_esc(area)}</td>"
                         f"<td style='border-bottom:1px solid #eee;padding:6px'>{_esc(a)}</td>"
                         f"<td style='border-bottom:1px solid #eee;padding:6px'>{_esc(rr)}</td></tr>")
            s.append("</tbody></table>")
        for f in r["findings"]:
            color, bg, tag = badge.get(f["level"], ("#444", "#f0f0f0", "•"))
            s.append(f"<div class='finding'>"
                     f"<span class='pill' style='color:{color};background:{bg}'>{tag}</span>"
                     f"{_md_bold(f['msg'])}</div>")
        primary = diag["coach"].get("primary")
        if primary:
            s.append("<h2>🎯 Start here</h2>")
            s.append(f"<p><strong>{_esc(primary['title'])}</strong></p>")
            s.append(f"<p><em>Why:</em> {_esc(primary['why'])}</p>")
            s.append(f"<p><em>Do this:</em> {_esc(primary['practice'])}</p>")

        rm = diag.get("roadmap")
        if rm:
            s.append("<h2>📈 The full roadmap</h2>")
            s.append("<p class='sub'>Your “start here” is the single most important "
                     "move. This is the whole path it sits on — do them in order. "
                     "✅ done · 🎯 you're here · ⬜ coming up.</p>")
            badge_rm = {"done": "✅", "now": "🎯", "next": "⬜"}
            for stage in rm:
                icon = badge_rm.get(stage["status"], "⬜")
                here = " — <strong>you're here</strong>" if stage["status"] == "now" else ""
                s.append(f"<h3 style='margin:18px 0 4px'>{icon} "
                         f"{_esc(stage['title'])}{here}</h3>")
                s.append(f"<p style='margin:2px 0'><em>{_esc(stage['what'])}</em></p>")
                s.append(f"<p style='margin:2px 0'><strong>Why it matters:</strong> "
                         f"{_esc(stage['why'])}</p>")
                s.append("<ul style='margin:4px 0'>"
                         + "".join(f"<li>{_esc(step)}</li>" for step in stage["steps"])
                         + "</ul>")

    if issues:
        s.append("<h2>🔟 Issues to work (IDS)</h2>")
        for it in issues:
            s.append(f"<div class='issue'><strong>{_esc(it['issue'])}</strong>"
                     f"<div class='meta'>Owner: {_esc(it['suggested_owner'])} · "
                     f"Next step: {_esc(it['next_step'])}</div></div>")

    s.append("<footer>Built by Team Doctor. AI drafted the charter and issues; the "
             "RACI and coach findings are deterministic rules — traceable, not "
             "guessed.<br>© 2026 Feifei Li. All rights reserved.</footer>")
    s.append("</body></html>")
    return "".join(s)
