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
    members = [Member.create((m.get("name") or "Unnamed").strip(),
                             (m.get("role") or "").strip())
               for m in spec.get("members", []) if m.get("name")]
    workstreams = [Workstream.create((w.get("name") or "Untitled").strip(),
                                     (w.get("description") or "").strip())
                   for w in spec.get("workstreams", []) if w.get("name")]

    member_id = {m.name.lower(): m.id for m in members}
    ws_id = {w.name.lower(): w.id for w in workstreams}

    raci: Dict[str, Dict[str, str]] = {w.id: {} for w in workstreams}
    for row in spec.get("raci", []):
        wid = ws_id.get(str(row.get("workstream", "")).strip().lower())
        mid = member_id.get(str(row.get("member", "")).strip().lower())
        code = str(row.get("code", "")).strip().upper()
        if wid and mid and code in ("A", "R", "C", "I"):
            raci.setdefault(wid, {})[mid] = code

    raci_result = raci_check.check(raci, workstreams, members)
    raci_errors = sum(1 for f in raci_result["findings"] if f["level"] == "error")

    coach = health.coach({
        "has_charter": bool(spec.get("mission")),
        "has_workstreams": bool(workstreams),
        "raci_errors": raci_errors,
        "decisions_count": 0,
        "responses_count": 0,
        "pulse": None,
        "has_rocks": False,
        "has_scorecard": False,
        "has_issues_resolved": False,
    })

    return {
        "members": members,
        "workstreams": workstreams,
        "raci": raci,
        "raci_result": raci_result,
        "coach": coach,
        "team_name": (spec.get("team_name") or "Your team").strip(),
        "mission": spec.get("mission", ""),
        "summary": spec.get("summary", ""),
    }


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
