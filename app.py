"""Team Doctor — a standalone, talk-to-it team health check.

Describe a team in plain English; an AI turns it into structure; a deterministic
engine diagnoses it. Self-explaining for a walk-up demo: type, click, get an
explainable health check. AI for understanding, rules for the verdict.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

from teamdoctor import doctor, llm

st.set_page_config(page_title="Team Doctor", page_icon="🩺", layout="wide")

st.title("🩺 Team Doctor")
st.caption("Describe your team in plain English — get an instant, explainable "
           "health check. AI understands what you wrote; the diagnosis itself is "
           "pure deterministic logic, so every flag is traceable, not guessed.")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("diagnosis", None)

# ── sidebar: pick any model ───────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ AI model")
    provider = st.selectbox("Provider", list(llm.PROVIDERS.keys()))
    cfg = llm.PROVIDERS[provider]
    model = st.text_input("Model", value=cfg["default_model"])

    api_key = ""
    if cfg.get("needs_key", True):
        prefill = ""
        try:
            if cfg.get("secret_key"):
                prefill = st.secrets.get(cfg["secret_key"], "")
        except Exception:
            prefill = ""
        api_key = st.text_input("API key", value=prefill, type="password")
        st.caption(f"🔑 Get a key: {cfg.get('get_key', '')}")
        if provider.startswith("Google") or provider.startswith("Groq"):
            st.caption("✅ This provider has a free tier.")
    else:
        st.caption("Runs locally — no key needed. Requires Ollama running "
                   "(`ollama serve`).")

    st.divider()
    st.caption("How it works: the AI turns your words into a team structure, then "
               "a deterministic engine (RACI + EOS coach) produces the diagnosis. "
               "The AI never invents a finding.")


def _need_key_missing() -> bool:
    return cfg.get("needs_key", True) and not api_key


def run_intake(user_text: str) -> None:
    """One user turn: extract+diagnose on the first pass, then grounded Q&A."""
    msgs = st.session_state.messages
    msgs.append({"role": "user", "content": user_text})
    try:
        if st.session_state.diagnosis is None:
            spec = doctor.extract(provider, model, api_key, msgs)
            if not spec.get("ready"):
                q = spec.get("follow_up") or ("Tell me a bit more — who's on the "
                                              "team and what are they working on?")
                msgs.append({"role": "assistant", "content": q})
            else:
                diag = doctor.diagnose(spec)
                st.session_state.diagnosis = diag
                msgs.append({"role": "assistant",
                             "content": doctor.summary_text(diag)})
        else:
            ans = doctor.answer(provider, model, api_key,
                                st.session_state.diagnosis, msgs)
            msgs.append({"role": "assistant", "content": ans})
    except llm.LLMError as e:
        msgs.append({"role": "assistant", "content": f"⚠️ {e}"})


def render_diagnosis(diag: dict) -> None:
    r = diag["raci_result"]
    if diag.get("summary"):
        st.caption(f"_What you described:_ {diag['summary']}")
    st.metric("RACI structure score", f"{round(r['score'] * 100)}%",
              help="Share of workstreams with exactly one owner and a doer.")
    for f in r["findings"]:
        icon = {"error": "🔴", "warn": "🟡", "ok": "🟢"}.get(f["level"], "•")
        st.markdown(f"{icon} {f['msg']}")

    primary = diag["coach"].get("primary")
    if primary:
        st.divider()
        st.markdown(f"### 🎯 Start here: {primary['title']}")
        st.markdown(f"**Why:** {primary['why']}")
        st.markdown(f"**Do this:** {primary['practice']}")
    also = diag["coach"].get("also", [])
    if also:
        with st.expander("Other things to watch"):
            for a in also:
                st.markdown(f"- {a}")
    st.caption(f"Maturity stage: **{diag['coach'].get('maturity', '—')}**  ·  "
               "Every flag above is a traceable rule — not AI guesswork.")

    st.download_button("📄 Download this report (.md)",
                       data=doctor.report_md(diag),
                       file_name=f"team-doctor-{diag['team_name'].lower().replace(' ', '-')}.md",
                       mime="text/markdown")


# ── quick actions ─────────────────────────────────────────────────────────────
b1, b2 = st.columns(2)
if b1.button("✨ Try a sample team", use_container_width=True):
    if _need_key_missing():
        st.warning("Add an API key in the sidebar first, or switch to Ollama.")
    else:
        with st.spinner("Diagnosing a sample team…"):
            run_intake(doctor.SAMPLE_TEAM)
        st.rerun()
if b2.button("🔄 Start over", use_container_width=True):
    st.session_state.messages = []
    st.session_state.diagnosis = None
    st.rerun()

# ── two-pane layout: conversation + diagnosis ─────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown("#### Conversation")
    if not st.session_state.messages:
        st.info("Describe your team below — for example: *“We're 4 people. Alex "
                "does almost everything, two people haven't shipped, and we never "
                "write down decisions.”*")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

with right:
    st.markdown("#### Diagnosis")
    if st.session_state.diagnosis:
        render_diagnosis(st.session_state.diagnosis)
    else:
        st.info("Your team's health check will appear here once you describe it.")

# ── chat input (always pinned to the bottom) ──────────────────────────────────
prompt = st.chat_input("Describe your team — or ask a follow-up question…")
if prompt:
    if _need_key_missing():
        st.warning("Add an API key in the sidebar first, or switch to Ollama.")
    else:
        with st.spinner("Thinking…"):
            run_intake(prompt)
        st.rerun()

st.divider()
st.caption("© 2026 Feifei Li. All rights reserved. Team Doctor is proprietary "
           "software. AI structures your description; the diagnosis is deterministic.")
