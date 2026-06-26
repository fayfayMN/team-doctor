"""Team Doctor — a standalone, talk-to-it team health check.

Describe a team in plain English; an AI turns it into structure; a deterministic
engine diagnoses it. Self-explaining for a walk-up demo: type, click, get an
explainable health check. AI for understanding, rules for the verdict.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from datetime import date, timedelta

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
    # Key the model field by provider so switching providers resets the model to
    # that provider's default. Without the per-provider key, Streamlit keeps the
    # old text (e.g. a Gemini model) and the call to Ollama would 404.
    model = st.text_input("Model", value=cfg["default_model"],
                          key=f"model::{provider}")

    api_key = ""
    if cfg.get("needs_key", True):
        # Bring-your-own-key only: the field always starts empty and the key is
        # never read from app secrets, so a visitor can never spend the owner's
        # quota. The key lives only in this browser session.
        api_key = st.text_input("Your API key", value="", type="password",
                                placeholder="paste your own key")
        st.caption(f"🔑 Get a free key: {cfg.get('get_key', '')}")
        st.caption("🔒 Your key is used only for this session, in your browser — "
                   "it's never stored or shared.")
        if provider.startswith("Google") or provider.startswith("Groq"):
            st.caption("✅ This provider has a free tier.")
    else:
        st.caption("Runs locally — no key needed. Requires Ollama running on the "
                   "same machine (only works when you run the app locally, not on "
                   "the cloud).")

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
    except Exception as e:
        # Never let an unexpected error end the run silently — show it instead of
        # crashing. Smaller local models sometimes return oddly-shaped JSON; this
        # turns that into a clear, recoverable message.
        msgs.append({"role": "assistant", "content":
                     "⚠️ Something went wrong turning that into a diagnosis "
                     f"(`{type(e).__name__}`). This often means the model returned "
                     "an unexpected shape — try again, shorten the description, or "
                     "switch to a stronger model (e.g. DeepSeek, or a bigger Ollama "
                     "model). The free Samples and Build-your-team paths always work."})


def render_charter(charter: dict) -> None:
    st.markdown("### 📜 Charter")
    if charter.get("mission"):
        st.markdown(f"**Mission:** {charter['mission']}")
    if charter.get("values"):
        st.markdown("**Values:** " + " · ".join(charter["values"]))
    rules = [("Decisions", charter.get("decision_rule")),
             ("Change & review window", charter.get("change_rule")),
             ("Communication", charter.get("communication_rule")),
             ("Credit", charter.get("credit_rule"))]
    for label, text in rules:
        if text:
            st.markdown(f"**{label}:** {text}")
    review = (date.today() + timedelta(days=90)).isoformat()
    st.caption(f"📅 Review this by {review} — put it on the calendar so it stays current.")


def _raci_table(diag: dict) -> list:
    """Build the actual ownership table behind the score, so the number is never
    shown without the content that produced it. Unowned areas show a proposed
    owner to make the gap actionable."""
    mname = {m.id: m.name for m in diag.get("members", [])}
    proposed = diag.get("proposed_owners", {})
    vac = diag.get("vacancies", {})
    rows = []
    for w in diag.get("workstreams", []):
        cell = diag.get("raci", {}).get(w.id, {})
        a = [mname.get(mid, mid) for mid, cs in cell.items() if "A" in cs]
        r = [mname.get(mid, mid) for mid, cs in cell.items() if "R" in cs]
        acc = ", ".join(a)
        if not acc:
            sug = proposed.get(w.id)
            if w.id in vac:
                acc = f"⚠️ VACANT — was {vac[w.id]['was']}"
                acc += f" · reassign to {sug}" if sug else " · reassign now"
            elif sug:
                acc = f"— none — · suggest: {sug}"
            else:
                acc = "— none —"
        rows.append({"Area of work": w.name,
                     "Accountable (owns it)": acc,
                     "Responsible (does it)": ", ".join(r) or "— none —"})
    return rows


def render_continuity(cont: dict) -> None:
    st.error(f"🚨 {cont['title']}")
    st.markdown(f"**Why:** {cont['why']}")
    for step in cont["steps"]:
        st.markdown(f"- {step}")
    st.divider()


def render_diagnosis(diag: dict) -> None:
    r = diag["raci_result"]
    # A recent collapse/resignation? Stabilizing comes before anything else.
    if diag.get("continuity"):
        render_continuity(diag["continuity"])

    st.markdown("### 🧩 Ownership (RACI)")
    if diag.get("summary"):
        st.caption(f"_What you described:_ {diag['summary']}")
    st.metric("RACI structure score", f"{round(r['score'] * 100)}%",
              help="Share of areas with exactly one owner and at least one doer. "
                   "This measures structural completeness only — risks like one "
                   "person owning too much are flagged separately below.")
    # Always show the table that produces the score — a number with no content
    # creates false confidence.
    table = _raci_table(diag)
    if table:
        st.table(table)
        cap = ("This is the ownership map behind the score. Gaps (— none —) and any "
               "one person owning several areas are exactly what to fix.")
        if diag.get("proposed_owners"):
            cap += (" “suggest:” names a proposed starting owner for each unowned "
                    "area — assign it now even temporarily, then validate with the team.")
        st.caption(cap)
    for f in r["findings"]:
        icon = {"error": "🔴", "warn": "🟡", "ok": "🟢"}.get(f["level"], "•")
        st.markdown(f"{icon} {f['msg']}")
    for note in diag.get("structure_notes", []):
        st.markdown(note)

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
    st.caption("Sorted by urgency: 🔴 act this week · 🟡 act this month · ⬜ plan ahead.")
    sev = {"urgent": "🔴 Urgent — act this week",
           "important": "🟡 Important — act this month",
           "planned": "⬜ Planned — act next term"}
    for it in issues:
        badge = sev.get(it.get("severity", "important"), sev["important"])
        st.markdown(f"{badge}")
        st.markdown(f"**{it['issue']}**")
        st.caption(f"Owner: {it['suggested_owner']} · Next step: {it['next_step']}")


def render_root_cause(ws: dict) -> None:
    """Show the root cause and any decision-authority conflict BEFORE prescriptions
    — fixing the cause matters more than the symptom list. AI path only."""
    rc = ws.get("root_cause")
    da = ws.get("decision_authority")
    trust = ws.get("trust")
    if not (rc or da or trust == "broken"):
        return
    st.markdown("### 🔍 What's really going on")
    if rc:
        st.markdown(f"**Root cause:** {rc}")
        st.caption("Everything below is downstream of this. Fix the cause, not just "
                   "the symptoms.")
    if da:
        models = da.get("models") or []
        st.markdown("**⚖️ You don't yet agree on how decisions get made.**")
        if models:
            st.markdown("Two views are in play: "
                        + " **vs.** ".join(f"*{m}*" for m in models))
        if da.get("first_step"):
            st.markdown(f"**Align on this first:** {da['first_step']}")
    if trust == "broken":
        st.warning("Trust here reads as **broken**, not just strained — so the fix is "
                   "forward-only governance (clear roles and decision rules), not a "
                   "reconciliation conversation. Build a structure that works even if "
                   "the relationship doesn't recover.")
    st.divider()


def render_workspace(ws: dict) -> None:
    render_root_cause(ws)
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
        ups = st.file_uploader(
            "…or upload one or more files — PDF, Word, text, or images "
            "(PNG/JPG, e.g. a photo of a whiteboard or org chart)",
            type=["pdf", "docx", "doc", "txt", "md",
                  "png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=True)
        if cfg.get("supports_vision"):
            st.caption("You can drop several at once. Images (PNG/JPG) are read by "
                       f"your selected vision model (**{provider}**).")
        else:
            st.caption(f"You can drop several at once. ⚠️ **{provider}** can't read "
                       "images — for a photo or screenshot, switch the model to "
                       "**Google Gemini** or **OpenAI** in the sidebar. Text, PDF, "
                       "and Word work with any model.")
        if st.button("🩺 Diagnose with AI", type="primary"):
            img_mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "webp": "image/webp", "gif": "image/gif"}
            parts, images = [], []
            if desc.strip():
                parts.append(desc.strip())
            for f in (ups or []):
                ext = (f.name.rsplit(".", 1)[-1] if "." in f.name else "").lower()
                if ext in img_mime:
                    images.append((img_mime[ext], f.read()))
                    continue
                try:
                    t = ingest.extract_text(f).strip()
                except Exception as e:
                    st.warning(f"Couldn't read {f.name}: {e}. Try a text PDF or .docx.")
                    continue
                if t and len(t) >= ingest.MIN_USABLE:
                    parts.append(t)
                else:
                    st.warning(f"⚠️ Got no readable text from **{f.name}** — it's "
                               "likely an image-only/scanned PDF. Re-save its pages as "
                               "PNG/JPG and upload those (your vision model will read "
                               "them), or paste the text directly. It was **not** "
                               "included in the diagnosis.")

            if not parts and not images:
                st.warning("Type or paste a description, or upload a file or image.")
            elif _need_key_missing():
                st.warning("Paste your own API key in the sidebar first "
                           "(or pick Ollama if running locally).")
            else:
                if images:
                    try:
                        with st.spinner(f"Reading {len(images)} image(s) with your "
                                        "vision model…"):
                            vtext = llm.transcribe_images(provider, model, api_key,
                                                          images).strip()
                        if vtext:
                            parts.append(vtext)
                    except llm.LLMError as e:
                        st.warning(str(e))
                combined = "\n\n".join(p for p in parts if p.strip())
                if not combined:
                    st.warning("Nothing readable to diagnose yet — try a clearer "
                               "image or paste a description.")
                else:
                    with st.spinner("The agent is reading your team…"):
                        run_intake(combined)
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

elif st.session_state.messages:
    # No diagnosis was produced, but the AI replied — e.g. it asked a follow-up
    # question or hit an error. Without this, that reply was saved but never shown,
    # so the page just looked like "spinner, then nothing."
    st.markdown("#### Conversation")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    st.caption("The AI needs a bit more to build a full diagnosis — reply below, or "
               "use a Sample / Build-your-team (no AI needed).")
    prompt = st.chat_input("Reply or add more detail…")
    if prompt:
        if _need_key_missing():
            st.warning("Paste your own API key in the sidebar to continue.")
        else:
            with st.spinner("Thinking…"):
                run_intake(prompt)
            st.rerun()

st.divider()
st.caption("© 2026 Feifei Li. All rights reserved. Team Doctor is proprietary "
           "software. The diagnosis is deterministic and free; AI (optional) only "
           "reads your description, using your own key.")
