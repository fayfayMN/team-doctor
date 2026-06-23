"""Team Doctor — a standalone, talk-to-it team health check.

Describe a team in plain English; an AI turns it into structure; a deterministic
engine diagnoses it. Self-explaining for a walk-up demo: type, click, get an
explainable health check. AI for understanding, rules for the verdict.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

from teamdoctor import agent, doctor, ingest, llm

st.set_page_config(page_title="Team Doctor", page_icon="🩺", layout="wide")

st.title("🩺 Team Doctor")
st.caption("Get an instant, explainable team health check — charter, ownership "
           "map, issues, and the one thing to fix next. The diagnosis is "
           "deterministic, so it's free and needs no API key. Optionally, let AI "
           "read a plain-English description (with your own free key).")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("workspace", None)
st.session_state.setdefault("skills", [])

# ── sidebar: pick any model ───────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ AI model (optional)")
    st.caption("Only needed for the plain-English option. Samples and the "
               "build-your-team form are free and need no key.")
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
    st.caption("How it works: the health check (RACI + EOS coach) is pure "
               "deterministic logic — free, no key, can't hallucinate. The AI "
               "option only turns a paragraph into structure, using your own key.")


def _need_key_missing() -> bool:
    return cfg.get("needs_key", True) and not api_key


def _has_content(ws) -> bool:
    return bool(ws) and any(ws.get(k) for k in ("charter", "diagnosis", "issues"))


def run_spec(spec: dict) -> None:
    """Run a structured spec through the deterministic engine — no AI, no key."""
    ws, skills = doctor.build_workspace(spec)
    st.session_state.workspace = ws
    st.session_state.skills = skills


def _clean_rows(rows, field):
    """Read names from a data_editor: drop blank/None cells, dedupe, keep order.

    Streamlit returns empty cells as None — without this, str(None) leaks a
    phantom 'None' entry into the team.
    """
    seen, out = set(), []
    for r in rows:
        v = r.get(field)
        v = (str(v) if v is not None else "").strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


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
        st.markdown("**Do this:**")
        st.markdown(primary["practice"])
    also = diag["coach"].get("also", [])
    if also:
        with st.expander("Other things to watch"):
            for a in also:
                st.markdown(f"- {a}")

    roadmap = diag.get("roadmap")
    if roadmap:
        with st.expander("📈 Want to go further? The full roadmap, step by step"):
            st.caption("Your “start here” above is the single most important move. "
                       "This is the whole path it sits on — do them in order. "
                       "✅ done · 🎯 you're here · ⬜ coming up.")
            badge = {"done": "✅", "now": "🎯", "next": "⬜"}
            for stage in roadmap:
                icon = badge.get(stage["status"], "⬜")
                here = " — **you're here**" if stage["status"] == "now" else ""
                st.markdown(f"#### {icon} {stage['title']}{here}")
                st.markdown(f"*{stage['what']}*")
                st.markdown(f"**Why it matters:** {stage['why']}")
                st.markdown("**How to do it:**")
                for step in stage["steps"]:
                    st.markdown(f"- {step}")
                st.markdown("")
    st.caption(f"Maturity stage: **{diag['coach'].get('maturity', '—')}**  ·  "
               "Every flag here is a traceable rule — not AI guesswork.")
    with st.expander("ℹ️ What do “RACI” and “EOS” mean?"):
        st.markdown(
            "**RACI** makes ownership clear for each area of work:\n"
            "- **A — Accountable:** the one person who owns the outcome\n"
            "- **R — Responsible:** who actually does the work\n"
            "- **C / I — Consulted / Informed:** who gives input / is kept in the loop\n\n"
            "**EOS (Entrepreneurial Operating System)** is a popular, lightweight way to "
            "run a small business or team (from the book *Traction*). The coach borrows "
            "its proven habits: one clear owner per function, a short weekly meeting to "
            "solve issues, a few quarterly priorities, and a simple scorecard.")


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


# ── input modes: two are free & key-less; AI is optional (bring your own key) ──
if not _has_content(st.session_state.workspace):
    mode = st.radio(
        "How do you want to start?",
        ["✨ Try a sample (free, no key)",
         "✍️ Build your team (free, no key)",
         "💬 Describe in plain English (uses your own AI key)"],
        index=0)

    # — Free path 1: pre-built sample teams, deterministic, no AI —
    if mode.startswith("✨"):
        s1, s2 = st.columns([3, 1])
        sample_name = s1.selectbox("Sample case study",
                                   list(doctor.SAMPLE_SPECS.keys()),
                                   label_visibility="collapsed")
        if s2.button("Run sample", type="primary", use_container_width=True):
            run_spec(doctor.SAMPLE_SPECS[sample_name])
            st.rerun()

    # — Free path 2: structured form, deterministic, no AI —
    elif mode.startswith("✍️"):
        st.caption("Add your people and the areas of work, then mark who owns what. "
                   "No AI, no key — the health check runs on rules.")

        # One-click example so visitors can see a filled-in team (and a real
        # diagnosis: a president owning everything + nobody owning finance).
        # Areas are ordered to match the ownership picks set below by index.
        EX_PEOPLE = [{"name": "Mia", "role": "President"},
                     {"name": "Sam", "role": "VP"},
                     {"name": "Jess", "role": "Treasurer"},
                     {"name": "Kim", "role": "Member"},
                     {"name": "Alex", "role": "Member"}]
        EX_AREAS = [{"workstream": "Events"}, {"workstream": "Recruiting"},
                    {"workstream": "Sponsorships"}, {"workstream": "Guest speakers"},
                    {"workstream": "Social media"}, {"workstream": "Finance"}]
        EX_ACC = ["Mia", "Mia", "Mia", "Mia", "Sam", "— none —"]
        EX_RES = [["Kim"], ["Sam"], ["Jess"], ["Kim"], ["Alex"], []]

        bcol1, bcol2 = st.columns([1, 1])
        if bcol1.button("✨ Fill with an example team"):
            st.session_state["td_example"] = True
            st.session_state["td_team"] = "Campus AI Club"
            st.session_state["td_mission"] = "Run great AI events and grow members"
            for i in range(len(EX_AREAS)):
                st.session_state[f"a_{i}"] = EX_ACC[i]
                st.session_state[f"r_{i}"] = EX_RES[i]
            st.rerun()
        if st.session_state.get("td_example") and bcol2.button("↩️ Clear the example"):
            for k in (["td_example", "td_team", "td_mission"]
                      + [f"a_{i}" for i in range(len(EX_AREAS))]
                      + [f"r_{i}" for i in range(len(EX_AREAS))]):
                st.session_state.pop(k, None)
            st.rerun()

        ex_on = st.session_state.get("td_example", False)
        # Changing the editor key when the example loads forces it to re-read the
        # example rows instead of keeping any prior edits.
        ed_key = "ex" if ex_on else "blank"

        team_name = st.text_input("Team name", key="td_team",
                                  placeholder="e.g. Acme, Robotics Club")
        mission = st.text_input("Mission (optional)", key="td_mission",
                                placeholder="one line — what the team is here to do")
        c_m, c_w = st.columns(2)
        with c_m:
            st.markdown("**People on the team**")
            st.caption("One row per person. Role is optional.")
            members_rows = st.data_editor(
                EX_PEOPLE if ex_on else [{"name": "", "role": ""}],
                num_rows="dynamic", key=f"m_ed_{ed_key}",
                use_container_width=True, hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Name"),
                    "role": st.column_config.TextColumn("Role (optional)"),
                })
        with c_w:
            st.markdown("**Areas of work**")
            st.caption("An *area of work* is a standing job the team is responsible "
                       "for — something that needs an owner even when no one's "
                       "actively working on it. Think **ongoing responsibilities, "
                       "not tasks or job titles.** One area per row, 1–3 words each.")
            with st.expander("ℹ️ How to pick good areas of work"):
                st.markdown(
                    "**The test:** if you can ask *“who owns this?”* and it should "
                    "have one clear answer, it's an area of work.\n\n"
                    "**Keep them:**\n"
                    "- **Ongoing**, not one-off — *Fundraising*, not *“book the venue”*\n"
                    "- **A function**, not a person — *Finance*, not *“Sam”*\n"
                    "- **Split when one person quietly does several** — if your "
                    "president runs recruiting, sponsors, builds, AND meetings, "
                    "that's **four** areas, not one. Splitting them is what makes "
                    "an overloaded owner show up in the diagnosis.\n\n"
                    "**Examples by team type:**\n"
                    "- **Startup:** Product · Sales · Marketing · Finance · Support · Hiring\n"
                    "- **Club:** Recruiting · Events · Fundraising · Social media · Sponsorships\n"
                    "- **Café:** Kitchen · Front of house · Ordering & supplies · Marketing · Finance\n"
                    "- **Nonprofit:** Programs · Grants · Volunteers · Outreach · Finance\n\n"
                    "Aim for roughly **4–8 areas** — enough to cover the real work, "
                    "few enough that each can have one owner.")
            ws_rows = st.data_editor(
                EX_AREAS if ex_on else [{"workstream": ""}],
                num_rows="dynamic", key=f"w_ed_{ed_key}",
                use_container_width=True, hide_index=True,
                column_config={
                    "workstream": st.column_config.TextColumn(
                        "Area of work",
                        help="A standing job the team owns — e.g. Sales, Product, "
                             "Finance, Marketing, Recruiting, Operations. A function, "
                             "not a one-off task or a person's name."),
                })

        member_names = _clean_rows(members_rows, "name")
        ws_names = _clean_rows(ws_rows, "workstream")

        if not (member_names and ws_names):
            st.info("Add at least one person and one area of work above — then a "
                    "dropdown appears to set who owns each area.")
        else:
            st.markdown("**Who owns what?** "
                        "(Accountable = owns it · Responsible = does it)")
            st.caption("Set an Accountable owner for each area, then click Run. "
                       "(Consulted/Informed aren't needed — they don't change "
                       "the diagnosis.)")
            picks = []
            for i, w in enumerate(ws_names):
                a_col, r_col = st.columns([2, 3])
                a = a_col.selectbox(f"Accountable · {w}",
                                    ["— none —"] + member_names, key=f"a_{i}")
                rs = r_col.multiselect(f"Responsible · {w}", member_names,
                                       key=f"r_{i}")
                picks.append((w, a, rs))
            if st.button("🩺 Run diagnosis", type="primary"):
                raci_rows = []
                for w, a, rs in picks:
                    if a != "— none —":
                        raci_rows.append({"workstream": w, "member": a, "code": "A"})
                    for r in rs:
                        raci_rows.append({"workstream": w, "member": r, "code": "R"})
                if not raci_rows:
                    st.warning("No owners assigned yet — use the **Accountable** "
                               "dropdowns above to set who owns each area, then run "
                               "again.")
                else:
                    run_spec({
                        "team_name": team_name or "Your team", "mission": mission,
                        "summary": "", "charter": None, "issues": None,
                        "members": [{"name": n} for n in member_names],
                        "workstreams": [{"name": w} for w in ws_names],
                        "raci": raci_rows})
                    st.rerun()
        st.caption("Want a drafted charter + an issues list too? Use the "
                   "plain-English option with your own free AI key.")

    # — Optional AI path: bring your own key —
    else:
        st.info("This option uses an AI model to read your description. Pick a "
                "provider and paste **your own** free key in the sidebar "
                "(e.g. Google Gemini — free at aistudio.google.com/apikey). "
                "Your key, your free quota — never the owner's.")
        desc = st.text_area(
            "Describe your team", height=140, label_visibility="collapsed",
            placeholder="“We're a 6-person startup. Our founder does sales, product, "
            "and support. Two engineers wait to be told what to do. Nobody owns "
            "finance. We argue about decisions and never write them down.”")
        up = st.file_uploader("…or upload a description (PDF, Word, .txt, .md)",
                              type=["pdf", "docx", "doc", "txt", "md"])
        if st.button("🩺 Diagnose with AI", type="primary"):
            text = desc.strip()
            if not text and up is not None:
                try:
                    text = ingest.extract_text(up).strip()
                except Exception as e:
                    text = ""
                    st.warning(f"Couldn't read that file: {e}. Try a text PDF or .docx.")
                if text and len(text) < ingest.MIN_USABLE:
                    st.warning("Couldn't extract readable text — try a text-based "
                               "PDF, a .docx, or paste the description.")
                    text = ""
            if not text:
                st.warning("Type or paste a description, or upload a file.")
            elif _need_key_missing():
                st.warning("Paste your own API key in the sidebar first "
                           "(or pick Ollama if running locally).")
            else:
                with st.spinner("The agent is reading your team…"):
                    run_intake(text)
                st.rerun()
else:
    if st.button("🔄 Start over / diagnose another team"):
        st.session_state.messages = []
        st.session_state.workspace = None
        st.session_state.skills = []
        st.rerun()

# ── display the result ────────────────────────────────────────────────────────
if _has_content(st.session_state.workspace):
    if st.session_state.messages:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Conversation")
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
        with right:
            st.markdown("#### Your operating system")
            render_workspace(st.session_state.workspace)
    else:
        st.markdown("#### Your operating system")
        render_workspace(st.session_state.workspace)

    if st.session_state.skills:
        with st.expander(f"🧩 What's in this report ({len(st.session_state.skills)})"):
            for s in st.session_state.skills:
                st.markdown(f"- {s}")

    # Follow-up chat only applies to the AI path (and uses the visitor's own key).
    if st.session_state.messages:
        prompt = st.chat_input("Ask a follow-up question about the diagnosis…")
        if prompt:
            if _need_key_missing():
                st.warning("Paste your own API key in the sidebar to ask follow-ups.")
            else:
                with st.spinner("Thinking…"):
                    run_intake(prompt)
                st.rerun()

st.divider()
st.caption("© 2026 Feifei Li. All rights reserved. Team Doctor is proprietary "
           "software. The diagnosis is deterministic and free; AI (optional) only "
           "reads your description, using your own key.")
