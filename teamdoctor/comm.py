"""Communication style — the deterministic trade-off engine.

Teams rarely argue about *whether* to communicate; they argue about *how* —
async vs. sync, one channel vs. scattered DMs, written vs. verbal. Each choice
has real, predictable consequences. This module names the four canonical styles,
their trade-offs, and — given a team's type, size, and whether it's in crisis —
recommends the best fit and (for Team Doctor) flags the style the team seems to
be running today.

No LLM: every style, trade-off, and recommendation is a transparent rule. The
same logic is mirrored in TeamUp's ``teamup/comm.py`` so both apps agree.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# The four styles, ordered best-record-first. Each carries the dimensions a team
# actually weighs, plus the one failure it's prone to. "ad_hoc" and pure
# "sync_meetings" are real and common, but never the recommended target.
STYLES: List[Dict] = [
    {"key": "async_channel",
     "name": "Async-first (one shared channel)",
     "speed": "Medium",
     "record": "Strong — written by default",
     "inclusion": "High — everyone sees the same thread",
     "scales": "Excellent",
     "watch_out": "Genuinely urgent calls can wait too long — pair it with a clear "
                  "escalation rule (tag the owner, then call)."},
    {"key": "hybrid",
     "name": "Hybrid (async default + a scheduled sync)",
     "speed": "Balanced",
     "record": "Strong, if you write decisions down",
     "inclusion": "High",
     "scales": "Good",
     "watch_out": "Takes discipline to log what was decided in the sync, not just "
                  "say it out loud and lose it."},
    {"key": "sync_meetings",
     "name": "Sync-heavy (meetings / calls)",
     "speed": "Fast in the room",
     "record": "Weak unless someone takes notes",
     "inclusion": "Only the people present",
     "scales": "Poor",
     "watch_out": "Anyone absent gets bypassed, and decisions evaporate the moment "
                  "the call ends."},
    {"key": "ad_hoc",
     "name": "Ad-hoc / DMs (no single channel)",
     "speed": "Feels fast",
     "record": "None",
     "inclusion": "Fragmented — different people know different things",
     "scales": "Breaks down quickly",
     "watch_out": "This is the #1 way teams quietly fall apart — decisions get made, "
                  "lost, and reversed in private with no trail."},
]

_BY_KEY: Dict[str, Dict] = {s["key"]: s for s in STYLES}

# Above this head-count, even a small-team default tips toward async-first: more
# people means a meeting can't include everyone and a written channel wins.
SMALL_TEAM_MAX = 6


def style(key: str) -> Dict:
    """Look up a style by key; falls back to hybrid for an unknown key."""
    return _BY_KEY.get(key, _BY_KEY["hybrid"])


def recommend(kind: str, size: int, in_crisis: bool = False) -> Tuple[str, str]:
    """Best-fit style for a team, with a plain-English 'why'.

    Deterministic and adapts to the three things that actually move the answer:
    team type, head-count, and whether the team is in flux. ad_hoc and pure
    sync_meetings are never recommended — they're the risks, not the target.
    """
    if in_crisis:
        return ("hybrid",
                "You're in flux — a short daily sync keeps everyone aligned, but log "
                "every decision in one shared channel so nothing gets reversed in "
                "private while things are moving fast.")
    if kind == "small business":
        return ("hybrid",
                "Shift-based, in-person teams do best with a quick pre-shift huddle "
                "plus one group chat for anything that comes up between shifts.")
    if kind == "startup" or size > SMALL_TEAM_MAX:
        return ("async_channel",
                "A busy or growing team needs an async-first channel that keeps a "
                "written record and doesn't depend on everyone being in the same "
                "meeting at the same time.")
    if kind == "nonprofit":
        return ("async_channel",
                "Part-time volunteers and board members can't all make one meeting — "
                "an async channel with a shared decisions doc keeps everyone included, "
                "with periodic syncs for the big calls.")
    # club / generic / small steady team
    return ("hybrid",
            "A small team runs best async-first with one regular sync — fast enough "
            "to decide, with a written trail so nothing gets lost between meetings.")


# ── Detect the team's CURRENT style from how they describe themselves ──────────
# Keyword heuristic only — returns None when the description gives no clear signal,
# so the app shows a recommendation without inventing a "current" diagnosis.
# Strong "no shared record" signals. Kept free of negation-prone words like
# "written down" (which appears in "nothing written down" and flips meaning) and
# bare "dm" (matches "admin"); we use "dms"/"direct message" instead.
_AD_HOC_HINTS = ("dms", "direct message", "texting", "texted", "privately",
                 "in private", "side channel", "side-channel", "verbal", "verbally",
                 "over text", "group text", "no record", "nothing written",
                 "off the record", "behind the scenes", "whoever they ask")
_SYNC_HINTS = ("meeting", "meetings", "call", "calls", "huddle", "in person",
               "in-person", "face to face", "face-to-face", "stand-up", "standup")
# Concrete channel/tool words only — not ambiguous phrases that can be negated.
_ASYNC_HINTS = ("slack", "discord", "teams channel", "one channel", "shared channel",
                "async", "asynchronous", "shared doc", "decisions doc",
                "notion", "wiki", "thread")


def detect_current(text: str) -> Optional[str]:
    """Infer how the team communicates today. None when there's no clear signal."""
    t = (text or "").lower()
    async_hit = any(w in t for w in _ASYNC_HINTS)
    sync_hit = any(w in t for w in _SYNC_HINTS)
    ad_hoc_hit = any(w in t for w in _AD_HOC_HINTS)
    # Ad-hoc is the most important to surface — it's the silent killer. But a team
    # that ALSO has a real channel/doc isn't purely ad-hoc, so require no async signal.
    if ad_hoc_hit and not async_hit:
        return "ad_hoc"
    if async_hit:
        # A channel/doc plus meetings is hybrid; a channel alone is async-first.
        return "hybrid" if sync_hit else "async_channel"
    if sync_hit:
        return "sync_meetings"
    return None


def assess(text: str, kind: str, size: int, in_crisis: bool = False) -> Dict:
    """Full communication-style read for a team: what they seem to do now, the
    risk in that, and what to do instead — all deterministic."""
    rec_key, why = recommend(kind, size, in_crisis)
    current = detect_current(text)
    current_risk = None
    if current and current != rec_key and current in ("ad_hoc", "sync_meetings"):
        current_risk = style(current)["watch_out"]
    return {
        "recommended": rec_key,
        "name": style(rec_key)["name"],
        "why": why,
        "current": current,
        "current_name": style(current)["name"] if current else None,
        "current_risk": current_risk,
        "styles": STYLES,
    }
