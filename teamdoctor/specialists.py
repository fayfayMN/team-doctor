"""Specialist sub-agents — one per Team OS layer.

Each specialist is a focused LLM call that drafts the artifact for its layer
(a charter, an issues list). The orchestrator in agent.py decides *when* to call
them; deterministic validation runs on what they return. So the agents draft, but
guardrails (value counts, owner/next-step completeness) are still rules.

This is the "LLM proposes, rules validate" doctrine applied per layer.
"""

from __future__ import annotations

from typing import Dict, List

from teamdoctor import llm

MAX_VALUES = 5
MAX_ISSUES = 6


def _user_text(conversation: List[Dict]) -> str:
    """The team's own words — what the user typed, joined."""
    return "\n".join(m["content"] for m in conversation if m.get("role") == "user")


def _team_brief(spec: Dict, conversation: List[Dict]) -> str:
    members = ", ".join(f"{m.get('name')} ({m.get('role','')})".strip()
                        for m in (spec or {}).get("members", [])) or "unknown"
    workstreams = ", ".join(w.get("name", "") for w in (spec or {}).get("workstreams", [])) or "unknown"
    return (f"Team: {(spec or {}).get('team_name', 'the team')}\n"
            f"Members: {members}\n"
            f"Workstreams: {workstreams}\n"
            f"What they said:\n{_user_text(conversation)}")


# ── Charter Agent (Foundation layer) ──────────────────────────────────────────

CHARTER_SYSTEM = """You are the Charter specialist for a team-health tool. Given a \
team's description and structure, draft a concise founding charter. Reply with \
ONE JSON object and nothing else:

{ "mission": string,              // one sentence, specific to what they do
  "values": [string],            // 3-5 short values (a word or short phrase each)
  "decision_rule": string,       // one sentence: how they make + commit to calls
  "communication_rule": string,  // one sentence: where/how they talk, response norms
  "credit_rule": string }        // one sentence: how contribution gets recognized

Be specific to this team — never generic boilerplate. Keep every field tight."""


def draft_charter(provider: str, model: str, api_key: str,
                  spec: Dict, conversation: List[Dict]) -> Dict:
    raw = llm.chat(provider, model, api_key,
                   [{"role": "system", "content": CHARTER_SYSTEM},
                    {"role": "user", "content": _team_brief(spec, conversation)}],
                   temperature=0.4, json_mode=True)
    data = llm.extract_json(raw) or {}
    values = [str(v).strip() for v in data.get("values", []) if str(v).strip()]
    return {
        "mission": str(data.get("mission", "")).strip(),
        "values": values[:MAX_VALUES],
        "decision_rule": str(data.get("decision_rule", "")).strip(),
        "communication_rule": str(data.get("communication_rule", "")).strip(),
        "credit_rule": str(data.get("credit_rule", "")).strip(),
    }


# ── Cadence Agent (Weekly rhythm — Level 10 / IDS) ────────────────────────────

ISSUES_SYSTEM = """You are the Cadence specialist for a team-health tool, trained \
on the EOS Level 10 / IDS method (Identify, Discuss, Solve). From the team's \
description, surface the real, solvable issues holding them back. Reply with ONE \
JSON object and nothing else:

{ "issues": [ { "issue": string,           // the problem, stated plainly
                "suggested_owner": string, // a named member who should own solving it
                "next_step": string } ] }  // one concrete first action

List 3-6 issues, most important first. Use ONLY problems implied by what they \
said. Suggested_owner must be one of the team members named; if unclear, use "TBD"."""


def surface_issues(provider: str, model: str, api_key: str,
                   spec: Dict, conversation: List[Dict]) -> List[Dict]:
    raw = llm.chat(provider, model, api_key,
                   [{"role": "system", "content": ISSUES_SYSTEM},
                    {"role": "user", "content": _team_brief(spec, conversation)}],
                   temperature=0.4, json_mode=True)
    data = llm.extract_json(raw) or {}
    issues: List[Dict] = []
    for it in data.get("issues", [])[:MAX_ISSUES]:
        title = str(it.get("issue", "")).strip()
        if not title:
            continue
        issues.append({
            "issue": title,
            "suggested_owner": str(it.get("suggested_owner", "TBD")).strip() or "TBD",
            "next_step": str(it.get("next_step", "")).strip(),
        })
    return issues


def issues_completeness(issues: List[Dict]) -> List[str]:
    """Deterministic IDS check: every issue needs a real owner and a next step."""
    notes: List[str] = []
    for it in issues:
        if it["suggested_owner"] == "TBD":
            notes.append(f"'{it['issue'][:40]}' has no owner yet — assign one.")
        if not it["next_step"]:
            notes.append(f"'{it['issue'][:40]}' has no next step — define one.")
    return notes
