"""The Team Doctor agent — one agent, several skills, one pass.

A single agent reads a team description and, in ONE LLM call, applies all its
skills at once: it structures the team, drafts a charter, and surfaces issues.
The deterministic engine (raci.check / health.coach) then runs locally — no extra
model calls — to produce the verdict.

Why one call instead of a multi-step loop: at a live booth, 4-6 sequential model
calls mean 10-15s of waiting. One call returns in a few seconds, is simpler, and
has fewer failure points. The agent still *shows the skills it applied*. And the
diagnosis is still deterministic, so it can't hallucinate a finding.

Provider-agnostic: the response is plain JSON, so the same agent works on Gemini,
Groq, DeepSeek, OpenAI, Claude, or Ollama.
"""

from __future__ import annotations

from typing import Dict, List

from teamdoctor import doctor, llm

AGENT_SYSTEM = """You are Team Doctor, a single agent with several skills. In ONE
response, read the user's team description and produce a complete operating-system
snapshot as ONE JSON object and NOTHING else:

{
  "ready": boolean,            // false ONLY if too vague to identify members/work
  "follow_up": string,         // one short question if not ready
  "team_name": string,
  "mission": string,           // from their words, or a crisp one you draft
  "summary": string,           // one sentence on what is going wrong
  "members": [{"name": string, "role": string}],
  "workstreams": [{"name": string, "description": string}],
  "raci": [{"workstream": string, "member": string, "code": "A"|"R"|"C"|"I"}],
  "charter": {
     "mission": string, "values": [string],
     "decision_rule": string, "communication_rule": string, "credit_rule": string },
  "issues": [{"issue": string, "suggested_owner": string, "next_step": string}]
}

The skills you apply in this single pass:
- Structure: extract members, workstreams, and ownership (RACI). A=Accountable
  (ideally one per workstream), R=Responsible (does the work), C=Consulted,
  I=Informed. Spell names exactly. Infer obvious ownership; never invent people or
  work they didn't mention.
- Charter: draft a tight, specific founding charter (3-5 short values; one-sentence
  rules). Not generic boilerplate.
- Issues: surface 3-6 real, solvable problems, most important first. suggested_owner
  must be a named member, or "TBD" if unclear; give one concrete next_step each.

Be faithful to what they said. If the description is too vague to identify the
members or their work, set ready=false and ask ONE concise follow_up."""


def run(provider: str, model: str, api_key: str,
        conversation: List[Dict]) -> Dict:
    """One agent, one call. Returns
    {type: "final"|"question", text, workspace, skills}."""
    messages = [{"role": "system", "content": AGENT_SYSTEM}] + list(conversation)
    raw = llm.chat(provider, model, api_key, messages,
                   temperature=0.3, json_mode=True)
    data = llm.extract_json(raw)
    if data is None:
        raise llm.LLMError("The model didn't return usable structure. Try "
                           "rephrasing, or switch to a stronger model.")

    ready = data.get("ready", True)
    has_team = bool(data.get("members")) or bool(data.get("workstreams"))
    if not ready or not has_team:
        q = data.get("follow_up") or ("Tell me a bit more — who's on the team and "
                                      "what is each person working on?")
        return {"type": "question", "text": q, "workspace": None, "skills": []}

    workspace, skills = doctor.build_workspace(data)
    text = (doctor.summary_text(workspace["diagnosis"])
            if workspace["diagnosis"] else "Here's what I found.")
    return {"type": "final", "text": text, "workspace": workspace, "skills": skills}
