"""The agent loop.

`run_agent` is a generator: it yields TraceEvent objects as it works, so the UI
can stream the agent's thinking live. It owns the control flow — which tool runs
when, when to ask for clarification, when to replan, when to fall back — while
`tools.py` owns the actual work.

Pipeline:
    parse -> geocode -> weather -> activities + food -> plan
          -> cost -> validate -> [replan once] -> [fallback] -> result
"""

from __future__ import annotations

import time
from typing import Iterator

import httpx

from agent import tools
from agent.llm import LLMClient, LLMError
from agent.models import (
    EventKind, Plan, PreferenceProfile, StepStatus, TraceEvent, Weather,
)
from data.osm import GeocodeError

_MAX_REPLANS = 1


def _step(tool: str, message: str) -> TraceEvent:
    return TraceEvent(kind=EventKind.STEP, tool=tool, status=StepStatus.RUNNING,
                      message=message)


def _done(tool: str, message: str, started: float,
          detail: str | None = None, status: StepStatus = StepStatus.DONE) -> TraceEvent:
    return TraceEvent(kind=EventKind.STEP, tool=tool, status=status, message=message,
                      detail=detail, elapsed_ms=round((time.perf_counter() - started) * 1000))


def _think(message: str) -> TraceEvent:
    return TraceEvent(kind=EventKind.THINKING, message=message)


def _error(message: str) -> TraceEvent:
    return TraceEvent(kind=EventKind.ERROR, message=message)


def run_agent(raw_input: str, llm: LLMClient, *,
              allow_clarify: bool = True) -> Iterator[TraceEvent]:
    """Run the full planning agent, streaming a trace of every step.

    Yields TraceEvents. Terminates on exactly one of:
      * a CLARIFY event (caller should collect answers and re-run),
      * an ERROR event (unrecoverable),
      * a RESULT event carrying the finished Plan.
    """
    # ---- Tool 1: parse preferences ---------------------------------------
    t0 = time.perf_counter()
    yield _step("parse_preferences", "Reading your request…")
    try:
        profile = tools.parse_preferences(raw_input, llm)
    except LLMError as exc:
        yield _error(f"Couldn't understand the request — the AI service failed: {exc}")
        return
    yield _done("parse_preferences",
                f"Understood: {profile.normalized_summary or profile.city}", t0,
                detail=f"{profile.city} · {profile.currency_symbol}{profile.budget:.0f} · "
                       f"{profile.available_hours}h · interests: "
                       f"{', '.join(profile.interests) or 'open'}")

    # ---- Clarifying questions (bonus) ------------------------------------
    if allow_clarify and profile.needs_clarification and profile.clarifying_questions:
        yield _think("Your request is a little open-ended — asking a couple of "
                     "quick questions to plan something you'll actually enjoy.")
        yield TraceEvent(kind=EventKind.CLARIFY, tool="parse_preferences",
                         status=StepStatus.DONE,
                         message="A couple of quick questions before I plan:",
                         questions=profile.clarifying_questions, profile=profile)
        return

    # ---- Tool 2: geocode -------------------------------------------------
    t0 = time.perf_counter()
    yield _step("geocode", f"Locating {profile.city} on the map…")
    try:
        geo = tools.geocode(profile.city)
    except GeocodeError as exc:
        yield _done("geocode", "City not found", t0, status=StepStatus.FAILED)
        yield _error(str(exc))
        return
    yield _done("geocode", f"Found {geo.display_name.split(',')[0]}", t0,
                detail=f"lat {geo.lat:.3f}, lon {geo.lon:.3f}")

    # ---- Tool 3: weather (best effort) -----------------------------------
    t0 = time.perf_counter()
    yield _step("get_weather", "Checking Saturday's forecast…")
    weather: Weather | None = tools.fetch_weather(geo)
    if weather:
        yield _done("get_weather",
                    f"{weather.description.title()}, "
                    f"{weather.temp_min_c:.0f}-{weather.temp_max_c:.0f}°C", t0,
                    detail=f"{weather.precipitation_chance}% rain on {weather.date}")
    else:
        yield _done("get_weather", "Forecast unavailable — planning without it", t0,
                    status=StepStatus.SKIPPED)

    # ---- Tools 4 & 5: venue discovery ------------------------------------
    t0 = time.perf_counter()
    yield _step("get_activity_options", "Searching OpenStreetMap for things to do…")
    try:
        activities = tools.find_activities(profile, geo)
    except httpx.HTTPError:
        activities = []
    yield _done("get_activity_options",
                f"{len(activities)} activity venue(s) found", t0,
                detail=", ".join(p.name for p in activities[:5]) or "none",
                status=StepStatus.DONE if activities else StepStatus.SKIPPED)

    t0 = time.perf_counter()
    yield _step("get_food_options", "Searching OpenStreetMap for places to eat…")
    try:
        food = tools.find_food(profile, geo)
    except httpx.HTTPError:
        food = []
    yield _done("get_food_options", f"{len(food)} food venue(s) found", t0,
                detail=", ".join(p.name for p in food[:5]) or "none",
                status=StepStatus.DONE if food else StepStatus.SKIPPED)

    if not activities and not food:
        yield _error(
            f"OpenStreetMap has no usable venues near {profile.city} for those "
            "interests. Try a larger nearby city or broader interests."
        )
        return

    # ---- Tools 6-8: plan, cost, validate, with one replan + fallback -----
    plan: Plan | None = None
    violations: list[str] = []

    for attempt in range(_MAX_REPLANS + 1):
        is_replan = attempt > 0
        t0 = time.perf_counter()
        label = "Re-planning to fix the issues…" if is_replan else "Designing your itinerary…"
        yield _step("generate_final_plan", label)
        try:
            draft, candidates = tools.generate_draft(
                profile, weather, activities, food, llm,
                violations=violations if is_replan else None,
            )
        except LLMError as exc:
            yield _error(f"The AI couldn't build a plan: {exc}")
            return
        items = tools.enrich_items(draft, candidates)
        yield _done("generate_final_plan", f"Drafted {len(items)} stop(s)", t0,
                    detail=draft.summary)

        # cost
        t0 = time.perf_counter()
        yield _step("estimate_cost", "Costing the plan and travel between stops…")
        cost = tools.estimate_cost(items, profile)
        yield _done("estimate_cost",
                    f"≈ {profile.currency_symbol}{cost.total_cost:.0f}, "
                    f"~{cost.total_minutes} min total", t0,
                    detail=" ".join(cost.notes))

        # validate
        t0 = time.perf_counter()
        yield _step("validate_plan", "Checking it against your budget and time…")
        result = tools.validate_plan(items, cost, profile, weather)
        if result.is_valid:
            yield _done("validate_plan", "Plan fits your budget and time ✓", t0)
            plan = _assemble(items, draft, cost, result, weather)
            break

        yield _done("validate_plan", "Plan needs adjustment", t0,
                    detail="; ".join(result.violations), status=StepStatus.FAILED)
        violations = result.violations
        if not is_replan:
            yield _think("The first draft broke a hard constraint — feeding the "
                         "problems back to the planner for a fix.")

    # ---- Fallback (bonus): nothing valid after the replan ----------------
    if plan is None:
        t0 = time.perf_counter()
        yield _think("Even the revised plan doesn't fit. Switching to a lean "
                     "fallback plan built around free and low-cost options.")
        yield _step("generate_final_plan", "Building a fallback plan…")
        try:
            draft, candidates = tools.generate_draft(
                profile, weather, activities, food, llm,
                violations=violations, fallback=True,
            )
        except LLMError as exc:
            yield _error(f"The AI couldn't build a fallback plan: {exc}")
            return
        items = tools.enrich_items(draft, candidates)
        cost = tools.estimate_cost(items, profile)
        result = tools.validate_plan(items, cost, profile, weather)
        # Surface any remaining over-budget/time issues as warnings, not errors.
        result.warnings = result.warnings + result.violations
        yield _done("generate_final_plan", f"Fallback plan ready ({len(items)} stops)", t0)
        plan = _assemble(items, draft, cost, result, weather, is_fallback=True)

    yield TraceEvent(kind=EventKind.RESULT, tool="generate_final_plan",
                     status=StepStatus.DONE, message="Your Saturday plan is ready.",
                     plan=plan, profile=profile)


def _assemble(items, draft, cost, validation, weather: Weather | None,
              *, is_fallback: bool = False) -> Plan:
    """Bundle the working pieces into the final Plan object."""
    weather_note = ""
    if weather:
        weather_note = (
            f"Forecast for {weather.date}: {weather.description}, "
            f"{weather.temp_min_c:.0f}-{weather.temp_max_c:.0f}°C, "
            f"{weather.precipitation_chance}% chance of rain."
        )
    return Plan(
        items=items, summary=draft.summary, tradeoffs=draft.tradeoffs,
        cost=cost, validation=validation, weather_note=weather_note,
        is_fallback=is_fallback,
    )
