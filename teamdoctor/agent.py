"""The Team Doctor agent — a real tool-using loop.

This is what makes Team Doctor *agentic* rather than a single prompt: the model
runs a think -> act -> observe loop. Each turn it emits ONE JSON action choosing
a tool; the loop executes that tool, feeds the result back as an observation, and
repeats until the agent calls `final_answer` (or asks the user a question).

The tools are the deterministic engine. So the agent decides *what to do*, but the
diagnosis itself — every flag, every recommendation — still comes from rules
(raci.check / health.coach). Agentic on the outside, explainable on the inside.

Provider-agnostic by design: actions are plain JSON, so the same loop works on
Gemini, Groq, DeepSeek, OpenAI, Claude, or Ollama without native function-calling.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from teamdoctor import doctor, llm

MAX_STEPS = 8

AGENT_SYSTEM = """You are Team Doctor, an agent that diagnoses team health.

You work in a loop: think, call ONE tool, observe the result, then repeat until
you can give a final answer. Reply with ONE JSON object each turn and NOTHING
else:

{ "thought": "one short sentence of reasoning",
  "tool": "set_team" | "run_health_check" | "ask_user" | "final_answer",
  "args": { ... } }

Tools:
- set_team — record the team structure from what the user described.
    args: { "team_name": str, "mission": str,
            "members": [{"name": str, "role": str}],
            "workstreams": [{"name": str, "description": str}],
            "raci": [{"workstream": str, "member": str, "code": "A"|"R"|"C"|"I"}] }
    RACI codes: A=Accountable (owns it, ideally one per workstream), R=Responsible
    (does the work), C=Consulted, I=Informed. Spell names exactly as listed.
    Infer obvious ownership; never invent people or work they didn't mention.
- run_health_check — run the deterministic engine on the recorded team. args: {}
    The observation returns the real findings and the recommended next step.
- ask_user — ask ONE clarifying question, only if you truly cannot identify the
    members or their work. args: { "question": str }
- final_answer — present the diagnosis warmly and concisely. args: { "message": str }

Normal flow: set_team -> run_health_check -> final_answer. Base your final_answer
ONLY on the health-check observation. Never state a finding the engine didn't."""


def run(provider: str, model: str, api_key: str, conversation: List[Dict],
        on_step: Optional[Callable[[Dict], None]] = None,
        max_steps: int = MAX_STEPS) -> Dict:
    """Run the agent loop over the conversation so far.

    Returns {type: "final"|"question", text, diagnosis|None, steps:[...]}.
    `on_step` (optional) is called with each step as it happens, for live UI.
    """
    spec: Optional[dict] = None
    diagnosis: Optional[dict] = None
    steps: List[Dict] = []
    messages = [{"role": "system", "content": AGENT_SYSTEM}] + list(conversation)

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
            spec = args
            obs = (f"Recorded '{args.get('team_name', 'team')}' — "
                   f"{len(args.get('members', []))} members, "
                   f"{len(args.get('workstreams', []))} workstreams, "
                   f"{len(args.get('raci', []))} ownership rows.")
            messages.append({"role": "user", "content": "OBSERVATION: " + obs})

        elif tool == "run_health_check":
            if not spec:
                messages.append({"role": "user",
                                 "content": "OBSERVATION: No team recorded yet — "
                                            "call set_team first."})
                continue
            diagnosis = doctor.diagnose(spec)
            messages.append({"role": "user",
                             "content": "OBSERVATION:\n"
                                        + doctor.findings_context(diagnosis)})

        elif tool == "ask_user":
            return {"type": "question",
                    "text": args.get("question",
                                     "Tell me a bit more about your team."),
                    "diagnosis": diagnosis, "steps": steps}

        elif tool == "final_answer":
            return {"type": "final", "text": args.get("message", ""),
                    "diagnosis": diagnosis, "steps": steps}

        else:
            messages.append({"role": "user",
                             "content": "OBSERVATION: Unknown tool — use one of "
                                        "the four listed tools."})

    # Safety net: if the loop ran out, present whatever we deterministically have.
    if spec and not diagnosis:
        diagnosis = doctor.diagnose(spec)
    text = (doctor.summary_text(diagnosis) if diagnosis
            else "I need a bit more detail about your team to run a check.")
    return {"type": "final", "text": text, "diagnosis": diagnosis, "steps": steps}
