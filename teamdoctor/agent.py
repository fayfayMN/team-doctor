"""The Team Doctor orchestrator — a multi-agent, tool-using loop.

This is what makes Team Doctor *agentic*: an orchestrator runs a think -> act ->
observe loop, choosing which specialist tool to call to build a team's operating
system one layer at a time (charter -> ownership -> issues), following the Team OS
maturity ladder.

The tools are the deterministic engine (raci.check / health.coach) and the
specialist sub-agents (specialists.py). The orchestrator decides *what to do*;
the diagnosis itself still comes from rules — agentic on the outside, explainable
on the inside.

Provider-agnostic by design: actions are plain JSON, so the same loop works on
Gemini, Groq, DeepSeek, OpenAI, Claude, or Ollama without native function-calling.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from teamdoctor import doctor, llm, specialists

MAX_STEPS = 10

AGENT_SYSTEM = """You are Team Doctor, an orchestrator agent that builds a team's
operating system one layer at a time by calling specialist tools.

You work in a loop: think, call ONE tool, observe the result, repeat — until you
call final_answer. Reply with ONE JSON object each turn and NOTHING else:

{ "thought": "one short sentence of reasoning",
  "tool": "set_team" | "draft_charter" | "run_health_check" | "surface_issues"
          | "ask_user" | "final_answer",
  "args": { ... } }

Tools (call roughly in this order — build the layers the team is missing):
- set_team — ALWAYS FIRST. Record the structure from what the user described.
    args: { "team_name": str, "mission": str,
            "members": [{"name": str, "role": str}],
            "workstreams": [{"name": str, "description": str}],
            "raci": [{"workstream": str, "member": str, "code": "A"|"R"|"C"|"I"}] }
    RACI: A=Accountable (owns it, ideally one per workstream), R=Responsible,
    C=Consulted, I=Informed. Use exact names. Infer obvious ownership; never invent.
- draft_charter — if the team has no clear mission/values/rules, draft a founding
    charter (Foundation layer). args: {}
- run_health_check — run the deterministic RACI + coach engine. args: {}
- surface_issues — list the team's real problems as solvable issues with an owner
    and a next step (Weekly rhythm / IDS layer). args: {}
- ask_user — ONE clarifying question, only if you truly cannot identify members or
    their work. args: { "question": str }
- final_answer — summarize what you built and the single most important next step.
    args: { "message": str }

Base your final_answer ONLY on tool observations. Never state a finding or fact a
tool didn't give you."""


def _empty_workspace() -> Dict:
    return {"spec": None, "diagnosis": None, "charter": None, "issues": None}


def run(provider: str, model: str, api_key: str, conversation: List[Dict],
        on_step: Optional[Callable[[Dict], None]] = None,
        max_steps: int = MAX_STEPS) -> Dict:
    """Run the orchestrator loop over the conversation so far.

    Returns {type: "final"|"question", text, workspace, steps:[...]}.
    `on_step` (optional) is called with each step as it happens, for live UI.
    """
    ws = _empty_workspace()
    steps: List[Dict] = []
    messages = [{"role": "system", "content": AGENT_SYSTEM}] + list(conversation)

    def observe(text: str) -> None:
        messages.append({"role": "user", "content": "OBSERVATION: " + text})

    for _ in range(max_steps):
        raw = llm.chat(provider, model, api_key, messages,
                       temperature=0.2, json_mode=True)
        action = llm.extract_json(raw)

        if not action or "tool" not in action:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "Return ONE valid JSON action object "
                                        "exactly as specified."})
            continue

        tool = action.get("tool", "")
        args = action.get("args", {}) or {}
        step = {"thought": action.get("thought", ""), "tool": tool, "args": args}
        steps.append(step)
        if on_step:
            on_step(step)
        messages.append({"role": "assistant", "content": json.dumps(action)})

        if tool == "set_team":
            ws["spec"] = args
            observe(f"Recorded '{args.get('team_name', 'team')}' — "
                    f"{len(args.get('members', []))} members, "
                    f"{len(args.get('workstreams', []))} workstreams, "
                    f"{len(args.get('raci', []))} ownership rows.")

        elif tool == "draft_charter":
            if not ws["spec"]:
                observe("No team recorded yet — call set_team first.")
                continue
            try:
                ws["charter"] = specialists.draft_charter(
                    provider, model, api_key, ws["spec"], conversation)
                # Feed the mission back so the coach knows a charter now exists.
                if ws["charter"].get("mission") and not ws["spec"].get("mission"):
                    ws["spec"]["mission"] = ws["charter"]["mission"]
                observe(f"Drafted a charter: mission set, "
                        f"{len(ws['charter']['values'])} values, decision/"
                        f"communication/credit rules.")
            except llm.LLMError as e:
                observe(f"Charter drafting failed: {e}")

        elif tool == "run_health_check":
            if not ws["spec"]:
                observe("No team recorded yet — call set_team first.")
                continue
            ws["diagnosis"] = doctor.diagnose(ws["spec"])
            observe(doctor.findings_context(ws["diagnosis"]))

        elif tool == "surface_issues":
            if not ws["spec"]:
                observe("No team recorded yet — call set_team first.")
                continue
            try:
                ws["issues"] = specialists.surface_issues(
                    provider, model, api_key, ws["spec"], conversation)
                gaps = specialists.issues_completeness(ws["issues"])
                note = f" {len(gaps)} need an owner/next step." if gaps else ""
                observe(f"Surfaced {len(ws['issues'])} issues with owners and "
                        f"next steps.{note}")
            except llm.LLMError as e:
                observe(f"Surfacing issues failed: {e}")

        elif tool == "ask_user":
            return {"type": "question",
                    "text": args.get("question",
                                     "Tell me a bit more about your team."),
                    "workspace": ws, "steps": steps}

        elif tool == "final_answer":
            return {"type": "final", "text": args.get("message", ""),
                    "workspace": ws, "steps": steps}

        else:
            observe("Unknown tool — use one of the listed tools.")

    # Safety net: present whatever we deterministically have.
    if ws["spec"] and not ws["diagnosis"]:
        ws["diagnosis"] = doctor.diagnose(ws["spec"])
    text = (doctor.summary_text(ws["diagnosis"]) if ws["diagnosis"]
            else "I need a bit more detail about your team to run a check.")
    return {"type": "final", "text": text, "workspace": ws, "steps": steps}
