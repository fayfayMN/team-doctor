# 🩺 Team Doctor

**Describe your team in plain English. Get an instant, explainable health check.**

Team Doctor is a talk-to-it front door for team health. You type how your team
actually works — who does what, what's going wrong — and an **AI agent** maps it,
runs a health check, and explains a structured diagnosis: who's overloaded, who
owns nothing, where accountability is broken, and the single best next step.

**Agentic on the outside, explainable on the inside.** The agent runs a real
think → act → observe loop, choosing tools to build the team and run the check.
But the tools *are* a deterministic rules engine (RACI checks + an EOS-style
coach), so the agent decides *what to do* while every actual finding comes from
logic — never hallucinated, always traceable.

It's a standalone companion to the **Team OS** app, focused on one thing: the
"walk up and try it" experience.

## How it works

1. **Describe** your team in the chat — or click *Try a sample team*.
2. The **agent** loops: it maps members/workstreams/ownership (RACI), then calls
   the deterministic health check as a tool, then writes the diagnosis. You can
   watch each step it took.
3. The engine flags structural problems and recommends one fix.
4. **Ask follow-ups** in plain language, or **download a report** to take away.

## Bring your own model

Pick any provider in the sidebar and paste a key — no code changes:

| Provider | Cost | Get a key |
|----------|------|-----------|
| **Google Gemini** | Free tier | https://aistudio.google.com/apikey |
| **Groq** | Free tier | https://console.groq.com/keys |
| DeepSeek | Very cheap | https://platform.deepseek.com/api_keys |
| OpenAI | Paid | https://platform.openai.com/api-keys |
| Anthropic (Claude) | Paid | https://console.anthropic.com/settings/keys |
| **Ollama** | Free, local | https://ollama.com/download (no key, runs offline) |

For a free, reliable demo: **Google Gemini** or **Groq**.

## Run locally

```bash
cd team-doctor
pip install -r requirements.txt
streamlit run app.py
```

Pick a provider in the sidebar, paste a key, and click **Try a sample team**.

## Design principles

- **Agent decides, logic verifies.** A tool-using agent drives the flow, but the
  diagnosis comes from deterministic RACI + coach rules. Nothing hallucinated.
- **Provider-agnostic agent.** Tool calls are plain JSON, so the same loop runs on
  Gemini, Groq, DeepSeek, OpenAI, Claude, or Ollama — no native function-calling
  required. Free tiers supported.
- **Self-explaining.** Built so a stranger can use it without narration; the
  agent's steps are shown so you can see how it reached the verdict.
- **No database, no login.** Stateless per session — describe a team, get a read.
