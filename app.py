"""Perfect Saturday Planner — Streamlit UI.

A hosted web app: type your city, budget, time, mood and interests, and an AI
agent researches real places (OpenStreetMap + Open-Meteo) and streams back a
personalised Saturday itinerary, with its reasoning shown live.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent.llm import LLMClient, LLMError
from agent.models import EventKind, Plan, StepStatus
from agent.orchestrator import run_agent

load_dotenv()

st.set_page_config(page_title="Perfect Saturday Planner", page_icon="🗓️",
                   layout="centered", initial_sidebar_state="collapsed")

CATEGORY_ICON = {"food": "🍽️", "activity": "🎯", "break": "☕", "travel": "🚕"}

INTEREST_OPTIONS = ["food", "music", "walks", "art", "history", "nature", "shopping",
                    "coffee", "sports", "nightlife", "photography", "books"]
CONSTRAINT_OPTIONS = ["vegetarian", "vegan", "avoid crowded places", "wheelchair accessible",
                      "no alcohol", "budget-conscious", "kid-friendly", "indoors only"]


# --------------------------------------------------------------------------
# API key
# --------------------------------------------------------------------------
def resolve_api_key() -> str | None:
    """Find the Gemini key: Streamlit secrets > env var > sidebar input."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # noqa: BLE001 - secrets file may not exist locally
        pass
    return os.getenv("GEMINI_API_KEY") or st.session_state.get("api_key_input")


# --------------------------------------------------------------------------
# Trace rendering
# --------------------------------------------------------------------------
def _elapsed(ms: int | None) -> str:
    if not ms:
        return ""
    return f" · {ms/1000:.1f}s" if ms >= 1000 else f" · {ms} ms"


def _step_line(ev) -> str:
    icon = {StepStatus.DONE: "✅", StepStatus.FAILED: "❌",
            StepStatus.SKIPPED: "⏭️"}.get(ev.status, "⏳")
    line = f"{icon} `{ev.tool}` — {ev.message}{_elapsed(ev.elapsed_ms)}"
    if ev.detail and ev.status != StepStatus.RUNNING:
        line += f"  \n&nbsp;&nbsp;&nbsp;<span style='color:#888;font-size:0.85em'>{ev.detail}</span>"
    return line


def render_trace_static(events: list) -> None:
    """Render a finished trace (skips transient 'running' lines)."""
    for ev in events:
        if ev.kind == EventKind.STEP and ev.status == StepStatus.RUNNING:
            continue
        if ev.kind == EventKind.STEP:
            st.markdown(_step_line(ev), unsafe_allow_html=True)
        elif ev.kind == EventKind.THINKING:
            st.markdown(f"💭 _{ev.message}_")
        elif ev.kind == EventKind.ERROR:
            st.markdown(f"❌ {ev.message}")


# --------------------------------------------------------------------------
# Plan rendering
# --------------------------------------------------------------------------
def render_plan(plan: Plan, profile) -> None:
    sym = profile.currency_symbol

    if plan.is_fallback:
        st.warning("⚠️ **Fallback plan** — your original budget/time were tight, "
                   "so this is a leaner day built around free and low-cost spots.")

    st.subheader("🗓️ Your Perfect Saturday")
    st.write(plan.summary)
    if plan.weather_note:
        st.caption(f"🌤️ {plan.weather_note}")

    # Headline metrics. Delta strings start with the signed number so Streamlit
    # colours them correctly (inverse: under budget / within time = green).
    c1, c2, c3 = st.columns(3)
    cost_delta = plan.cost.total_cost - profile.budget
    time_delta_h = (plan.cost.total_minutes - profile.available_hours * 60) / 60
    c1.metric("Estimated cost", f"{sym}{plan.cost.total_cost:.0f}",
              delta=f"{cost_delta:+,.0f} vs budget", delta_color="inverse",
              help="Negative means the plan is under budget.")
    c2.metric("Time needed", f"{plan.cost.total_minutes/60:.1f} h",
              delta=f"{time_delta_h:+.1f} h vs free time", delta_color="inverse",
              help="Negative means the plan fits inside your available time.")
    c3.metric("Stops", str(len(plan.items)))

    budget_used = min(plan.cost.total_cost / profile.budget, 1.0) if profile.budget else 0
    st.progress(budget_used, text=f"Budget used: {sym}{plan.cost.total_cost:.0f} "
                                  f"of {sym}{profile.budget:.0f}")

    # Anti-hallucination indicator.
    g = plan.grounding
    if g.venue_stops and g.grounding_score >= 1.0:
        st.success(f"🔒 Grounded: all {g.venue_stops} venues are real, verified "
                   "OpenStreetMap places — nothing invented.")
    elif g.venue_stops:
        st.warning(f"🔒 Grounded: {g.grounded_stops}/{g.venue_stops} venues verified "
                   "on OpenStreetMap — unverified stops are flagged below.")

    # Itinerary timeline.
    st.markdown("#### The itinerary")
    for item in plan.items:
        icon = CATEGORY_ICON.get(item.category, "📍")
        with st.container(border=True):
            top, cost_col = st.columns([4, 1])
            with top:
                st.markdown(f"**{item.start_time} · {icon} {item.title}**")
                if item.location_name and item.location_name != item.title:
                    st.caption(f"📍 {item.location_name}")
            with cost_col:
                label = "Free" if item.estimated_cost <= 0 else f"{sym}{item.estimated_cost:.0f}"
                st.markdown(f"<div style='text-align:right'><b>{label}</b><br>"
                            f"<span style='color:#888;font-size:0.8em'>"
                            f"{item.duration_minutes} min</span></div>",
                            unsafe_allow_html=True)
            st.markdown(f"💡 _{item.why_it_fits}_")
            if item.osm_id:
                st.caption(f"✅ Verified real place — "
                           f"[view on OpenStreetMap](https://www.openstreetmap.org/{item.osm_id})")
            elif item.category in ("food", "activity"):
                st.caption("⚠️ Not tied to a verified place — treat as a loose suggestion.")

    # Cost breakdown.
    with st.expander("💰 Cost breakdown"):
        b = plan.cost
        st.markdown(
            f"- Activities: **{sym}{b.activities_cost:.0f}**\n"
            f"- Food: **{sym}{b.food_cost:.0f}**\n"
            f"- Travel (~{b.travel_minutes} min between stops): **{sym}{b.travel_cost:.0f}**\n"
            f"- **Total: {sym}{b.total_cost:.0f}**"
        )
        for note in b.notes:
            st.caption(note)

    # Trade-offs (bonus: honest compromises).
    if plan.tradeoffs:
        st.markdown("#### ⚖️ Trade-offs the agent made")
        for t in plan.tradeoffs:
            st.markdown(f"- {t}")

    # Soft warnings.
    for w in plan.validation.warnings:
        st.warning(w)


# --------------------------------------------------------------------------
# Run the agent (live streaming)
# --------------------------------------------------------------------------
def run_and_stream(raw_input: str, allow_clarify: bool) -> None:
    """Execute the agent, streaming the trace, and store the outcome."""
    api_key = resolve_api_key()
    if not api_key:
        st.session_state.mode = "result"
        st.session_state.error = ("No Gemini API key configured. Add it in the "
                                  "sidebar, or in Streamlit secrets when deployed.")
        st.rerun()

    try:
        llm = LLMClient(api_key=api_key)
    except LLMError as exc:
        st.session_state.mode = "result"
        st.session_state.error = str(exc)
        st.rerun()

    st.markdown("#### 🤖 The agent is working…")
    trace_box = st.container()
    events: list = []
    current_ph = None

    with trace_box:
        try:
            for ev in run_agent(raw_input, llm, allow_clarify=allow_clarify):
                events.append(ev)
                if ev.kind == EventKind.STEP and ev.status == StepStatus.RUNNING:
                    current_ph = st.empty()
                    current_ph.markdown(_step_line(ev), unsafe_allow_html=True)
                elif ev.kind == EventKind.STEP:
                    (current_ph or st.empty()).markdown(_step_line(ev),
                                                        unsafe_allow_html=True)
                    current_ph = None
                elif ev.kind == EventKind.THINKING:
                    st.markdown(f"💭 _{ev.message}_")
                    current_ph = None
                elif ev.kind == EventKind.CLARIFY:
                    st.session_state.mode = "clarify"
                    st.session_state.clarify_questions = ev.questions
                    st.session_state.clarify_base = raw_input
                    st.rerun()
                elif ev.kind == EventKind.ERROR:
                    st.session_state.error = ev.message
                elif ev.kind == EventKind.RESULT:
                    st.session_state.plan = ev.plan
                    st.session_state.profile = ev.profile
        except LLMError as exc:
            st.session_state.error = f"AI service error: {exc}"
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            st.session_state.error = f"Something went wrong: {exc}"

    st.session_state.trace = events
    st.session_state.mode = "result"
    st.rerun()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {"mode": "input", "raw_input": "", "allow_clarify": True,
                "clarify_questions": [], "clarify_base": "", "plan": None,
                "profile": None, "trace": [], "error": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset() -> None:
    for k in ("plan", "profile", "trace", "error", "clarify_questions"):
        st.session_state[k] = [] if k in ("trace", "clarify_questions") else None
    st.session_state.mode = "input"


init_state()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    if resolve_api_key():
        st.success("Gemini API key detected.")
    else:
        st.text_input("Gemini API key", type="password", key="api_key_input",
                      help="Get a free key at aistudio.google.com/apikey")
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "An AI agent runs 8 tools — it parses your request, geocodes the city, "
        "checks the forecast, searches **real venues on OpenStreetMap**, drafts "
        "an itinerary, costs it, validates it against your budget/time, and "
        "replans if needed.\n\n"
        "No place data is hardcoded — it works for any city."
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🗓️ Perfect Saturday Planner")
st.caption("Tell the agent about your ideal Saturday — it researches real places "
           "and builds a plan that actually fits you.")

# --------------------------------------------------------------------------
# Mode: INPUT
# --------------------------------------------------------------------------
if st.session_state.mode == "input":
    tab_form, tab_text = st.tabs(["📝 Quick form", "💬 Free text"])

    with tab_form:
        with st.form("plan_form"):
            city = st.text_input("City", placeholder="e.g. Bangalore")
            col1, col2 = st.columns(2)
            budget = col1.number_input("Budget", min_value=0, value=2000, step=100,
                                       help="In your city's local currency.")
            hours = col2.number_input("Hours free", min_value=1.0, max_value=16.0,
                                      value=4.0, step=0.5)
            mood = st.text_input("Mood", placeholder="e.g. tired but want something fun")
            interests = st.multiselect("Interests", INTEREST_OPTIONS,
                                       default=["food", "walks"])
            extra_interests = st.text_input("Other interests (comma-separated)",
                                            placeholder="e.g. street food, live jazz")
            constraints = st.multiselect("Constraints", CONSTRAINT_OPTIONS)
            extra_constraints = st.text_input("Other constraints (comma-separated)",
                                              placeholder="e.g. back home by 6pm")
            start_pref = st.selectbox("Preferred start",
                                      ["flexible", "morning", "afternoon", "evening"])
            submitted = st.form_submit_button("✨ Plan my Saturday", type="primary",
                                              use_container_width=True)

        if submitted:
            if not city.strip():
                st.error("Please enter a city.")
            else:
                all_interests = interests + [s.strip() for s in extra_interests.split(",") if s.strip()]
                all_constraints = constraints + [s.strip() for s in extra_constraints.split(",") if s.strip()]
                st.session_state.raw_input = (
                    f"City: {city}\nBudget: {budget}\nAvailable time: {hours} hours\n"
                    f"Preferred start: {start_pref}\nMood: {mood or 'not specified'}\n"
                    f"Interests: {', '.join(all_interests) or 'open to anything'}\n"
                    f"Constraints: {', '.join(all_constraints) or 'none'}"
                )
                st.session_state.allow_clarify = True
                st.session_state.mode = "running"
                st.rerun()

    with tab_text:
        st.write("Describe your ideal Saturday in your own words:")
        free_text = st.text_area("Your request", height=140,
                                 placeholder="I'm in Bangalore with about ₹2000 and "
                                 "4 free hours on Saturday. Feeling a bit tired but "
                                 "want something fun — I love food, music and walks. "
                                 "Vegetarian, and please avoid crowded places.",
                                 label_visibility="collapsed")
        if st.button("✨ Plan my Saturday", type="primary", key="text_submit",
                     use_container_width=True):
            if not free_text.strip():
                st.error("Please describe what you're looking for.")
            else:
                st.session_state.raw_input = free_text.strip()
                st.session_state.allow_clarify = True
                st.session_state.mode = "running"
                st.rerun()

# --------------------------------------------------------------------------
# Mode: RUNNING (streams the agent trace live)
# --------------------------------------------------------------------------
elif st.session_state.mode == "running":
    run_and_stream(st.session_state.raw_input, st.session_state.allow_clarify)

# --------------------------------------------------------------------------
# Mode: CLARIFY (bonus: agent asks 1-2 questions)
# --------------------------------------------------------------------------
elif st.session_state.mode == "clarify":
    st.info("🤔 The agent needs a little more to plan something you'll love:")
    with st.form("clarify_form"):
        answers = []
        for i, q in enumerate(st.session_state.clarify_questions):
            answers.append(st.text_input(q, key=f"clarify_{i}"))
        cont = st.form_submit_button("Continue planning →", type="primary")
    if cont:
        qa = "\n".join(f"Q: {q}\nA: {a or 'no preference'}"
                       for q, a in zip(st.session_state.clarify_questions, answers))
        st.session_state.raw_input = (st.session_state.clarify_base
                                      + "\n\nClarifying answers:\n" + qa)
        st.session_state.allow_clarify = False
        st.session_state.mode = "running"
        st.rerun()
    if st.button("Skip — just plan something"):
        st.session_state.raw_input = st.session_state.clarify_base
        st.session_state.allow_clarify = False
        st.session_state.mode = "running"
        st.rerun()

# --------------------------------------------------------------------------
# Mode: RESULT
# --------------------------------------------------------------------------
elif st.session_state.mode == "result":
    if st.session_state.error and not st.session_state.plan:
        st.error(f"😕 {st.session_state.error}")
    elif st.session_state.plan:
        render_plan(st.session_state.plan, st.session_state.profile)

    if st.session_state.trace:
        with st.expander("🔍 Agent trace — what it did, step by step",
                         expanded=not st.session_state.plan):
            render_trace_static(st.session_state.trace)

    st.markdown("---")
    if st.button("← Plan another Saturday", use_container_width=True):
        reset()
        st.rerun()
