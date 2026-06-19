# 🩺 Team Doctor

**Describe your team in plain English. Get an instant, explainable health check.**

Team Doctor is a talk-to-it front door for team health. You type how your team
actually works — who does what, what's going wrong — and it returns a structured
diagnosis: who's overloaded, who owns nothing, where accountability is broken,
and the single best next step.

**AI for understanding, deterministic logic for the verdict.** A language model
turns your messy description into structure; a rules engine (RACI checks + an
EOS-style coach) produces every finding. The AI never invents a problem — so the
diagnosis is always traceable.

It's a standalone companion to the **Team OS** app, focused on one thing: the
"walk up and try it" experience.

## How it works

1. **Describe** your team in the chat — or click *Try a sample team*.
2. The AI extracts members, workstreams, and ownership (RACI).
3. The deterministic engine flags structural problems and recommends one fix.
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

- **AI structures, logic decides.** The model only parses language and narrates;
  every finding comes from deterministic RACI + coach rules. Nothing hallucinated.
- **Self-explaining.** Built so a stranger can use it without narration.
- **Provider-agnostic.** Swap models freely; free tiers supported.
- **No database, no login.** Stateless per session — describe a team, get a read.
