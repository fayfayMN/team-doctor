"""Team Doctor — a standalone, talk-to-it team health check.

Describe a team in plain English; an AI turns it into structure; a deterministic
engine diagnoses it. Self-explaining for a walk-up demo: type, click, get an
explainable health check. AI for understanding, rules for the verdict.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

from teamdoctor import agent, doctor, llm

st.set_page_config(page_title="Team Doctor", page_icon="🩺", layout="wide")

st.title("🩺 Team Doctor")
st.caption("Describe your team in plain English — one AI agent applies several "
           "skills in a single pass: it drafts your charter, maps ownership, runs "
           "a health check, and surfaces your issues. The agent drafts; the "
           "diagnosis comes from deterministic rules, so every flag is traceable.")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("workspace", None)
st.session_state.setdefault("skills", [])

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
    st.caption("How it works: one agent applies several skills in a single pass — "
               "structure, charter, ownership, issues. The RACI + coach verdict is "
               "deterministic, so the agent never invents a finding.")


def _need_key_missing() -> bool:
    return cfg.get("needs_key", True) and not api_key


def _has_content(ws) -> bool:
    return bool(ws) and any(ws.get(k) for k in ("charter", "diagnosis", "issues"))


def run_intake(user_text: str) -> None:
    """One user turn. First pass runs the orchestrator (think -> act -> observe);
    once a workspace exists, follow-ups are grounded Q&A on everything built."""
    msgs = st.session_state.messages
    msgs.append({"role": "user", "content": user_text})
    try:
        if not _has_content(st.session_state.workspace):
            result = agent.run(provider, model, api_key, msgs)
            st.session_state.skills = result.get("skills", [])
            if _has_content(result.get("workspace")):
                st.session_state.workspace = result["workspace"]
            msgs.append({"role": "assistant", "content": result["text"]})
        else:
            ans = doctor.answer(provider, model, api_key,
                                st.session_state.workspace, msgs)
            msgs.append({"role": "assistant", "content": ans})
    except llm.LLMError as e:
        msgs.append({"role": "assistant", "content": f"⚠️ {e}"})


def render_charter(charter: dict) -> None:
    st.markdown("### 📜 Charter")
    if charter.get("mission"):
        st.markdown(f"**Mission:** {charter['mission']}")
    if charter.get("values"):
        st.markdown("**Values:** " + " · ".join(charter["values"]))
    rules = [("Decisions", charter.get("decision_rule")),
             ("Communication", charter.get("communication_rule")),
             ("Credit", charter.get("credit_rule"))]
    for label, text in rules:
        if text:
            st.markdown(f"**{label}:** {text}")


def render_diagnosis(diag: dict) -> None:
    r = diag["raci_result"]
    st.markdown("### 🧩 Ownership (RACI)")
    if diag.get("summary"):
        st.caption(f"_What you described:_ {diag['summary']}")
    st.metric("RACI structure score", f"{round(r['score'] * 100)}%",
              help="Share of workstreams with exactly one owner and a doer.")
    for f in r["findings"]:
        icon = {"error": "🔴", "warn": "🟡", "ok": "🟢"}.get(f["level"], "•")
        st.markdown(f"{icon} {f['msg']}")

    primary = diag["coach"].get("primary")
    if primary:
        st.markdown(f"#### 🎯 Start here: {primary['title']}")
        st.markdown(f"**Why:** {primary['why']}")
        st.markdown(f"**Do this:** {primary['practice']}")
    also = diag["coach"].get("also", [])
    if also:
        with st.expander("Other things to watch"):
            for a in also:
                st.markdown(f"- {a}")
    st.caption(f"Maturity stage: **{diag['coach'].get('maturity', '—')}**  ·  "
               "Every flag here is a traceable rule — not AI guesswork.")


def render_issues(issues: list) -> None:
    st.markdown("### 🔟 Issues to work (IDS)")
    for it in issues:
        st.markdown(f"**{it['issue']}**")
        st.caption(f"Owner: {it['suggested_owner']} · Next step: {it['next_step']}")


def render_workspace(ws: dict) -> None:
    if ws.get("charter"):
        render_charter(ws["charter"])
        st.divider()
    if ws.get("diagnosis"):
        render_diagnosis(ws["diagnosis"])
        st.divider()
    if ws.get("issues"):
        render_issues(ws["issues"])
        st.divider()
    team = (ws.get("diagnosis") or {}).get("team_name", "team")
    slug = team.lower().replace(" ", "-")
    st.download_button("📄 Download this plan (HTML — opens in browser, print to PDF)",
                       data=doctor.report_html(ws),
                       file_name=f"team-doctor-{slug}.html",
                       mime="text/html")
    st.caption("Tip: open the file and use your browser's **Print → Save as PDF** "
               "for a polished PDF.")


# ── quick actions ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 1, 1])
sample_name = c1.selectbox("Sample case study", list(doctor.SAMPLES.keys()),
                           label_visibility="collapsed")
if c2.button("✨ Run sample", use_container_width=True):
    if _need_key_missing():
        st.warning("Add an API key in the sidebar first, or switch to Ollama.")
    else:
        with st.spinner("The agents are building this team's operating system…"):
            run_intake(doctor.SAMPLES[sample_name])
        st.rerun()
if c3.button("🔄 Start over", use_container_width=True):
    st.session_state.messages = []
    st.session_state.workspace = None
    st.session_state.skills = []
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

    if st.session_state.skills:
        with st.expander(f"🧠 Skills the agent used ({len(st.session_state.skills)})",
                         expanded=True):
            for s in st.session_state.skills:
                st.markdown(f"- {s}")

with right:
    st.markdown("#### Your operating system")
    if _has_content(st.session_state.workspace):
        render_workspace(st.session_state.workspace)
    else:
        st.info("Your charter, ownership check, and issues will appear here once "
                "you describe your team.")

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
