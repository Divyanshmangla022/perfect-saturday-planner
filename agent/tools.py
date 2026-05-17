"""The agent's tools.

Each function here is one discrete, testable capability. The orchestrator wires
them together; none of them know about the others. Three of them (parse,
generate) use the LLM; the rest are deterministic code over real API data.
"""

from __future__ import annotations

import math

import httpx

from agent import prompts
from agent.llm import LLMClient
from agent.models import (
    CostBreakdown, DraftPlan, GeoLocation, GroundingReport, PlanItem, POI,
    PreferenceProfile, ValidationResult, Weather,
)
from data import osm, weather as weather_api

# Average urban travel speed (km/h) including traffic — used to turn the real
# distance between two venues into a realistic travel time.
_CITY_SPEED_KMH = 18.0

# Input guardrail: cap raw input length to bound cost and blunt prompt-injection
# payloads. A genuine request never needs this much text.
MAX_INPUT_CHARS = 4000

# ISO 4217 currency code -> display symbol. The LLM reliably produces ASCII
# currency codes but often mangles non-ASCII symbol glyphs (emitting a unicode
# escape sequence as literal text instead of the glyph), so we map it ourselves.
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "INR": "₹",
    "AUD": "A$", "CAD": "C$", "NZD": "NZ$", "HKD": "HK$", "SGD": "S$",
    "CHF": "CHF", "SEK": "kr", "NOK": "kr", "DKK": "kr", "PLN": "zł",
    "RUB": "₽", "TRY": "₺", "BRL": "R$", "MXN": "$", "ZAR": "R", "AED": "AED",
    "SAR": "SAR", "ILS": "₪", "THB": "฿", "KRW": "₩", "IDR": "Rp", "MYR": "RM",
    "PHP": "₱", "VND": "₫", "NPR": "₨", "PKR": "₨", "LKR": "₨", "BDT": "৳",
    "EGP": "E£",
}


def _currency_symbol(code: str) -> tuple[str, str]:
    """Return a clean (ISO code, display symbol) pair from a raw LLM value."""
    code = (code or "").strip().upper()
    if not (len(code) == 3 and code.isalpha()):
        code = "USD"
    return code, CURRENCY_SYMBOLS.get(code, f"{code} ")


class InputRejected(Exception):
    """Raised by the input guardrail for oversized input."""


# ==========================================================================
# Tool 1 — parse the user's request into a structured profile  (LLM)
# ==========================================================================
def parse_preferences(raw_input: str, llm: LLMClient) -> PreferenceProfile:
    """Turn a form dump or free-text paragraph into a PreferenceProfile.

    Applies two input guardrails: a hard length cap (here) and a content/scope
    check (delegated to the LLM via is_plannable / rejection_reason).
    """
    raw_input = (raw_input or "").strip()
    if not raw_input:
        raise InputRejected("Please describe what kind of Saturday you'd like.")
    if len(raw_input) > MAX_INPUT_CHARS:
        raise InputRejected(
            f"That request is very long ({len(raw_input)} characters). "
            f"Please keep it under {MAX_INPUT_CHARS} characters."
        )
    prompt = prompts.PARSE_USER_TEMPLATE.format(raw_input=raw_input)
    profile = llm.complete_json(
        prompt, PreferenceProfile, system=prompts.PARSE_SYSTEM, temperature=0.2,
    )
    # Derive the currency symbol from the ISO code in code — never trust the
    # LLM to emit the non-ASCII glyph itself.
    profile.currency, profile.currency_symbol = _currency_symbol(profile.currency)
    # Guard against an over-eager model asking endless questions.
    profile.clarifying_questions = profile.clarifying_questions[:2]
    if not profile.clarifying_questions:
        profile.needs_clarification = False
    return profile


# ==========================================================================
# Tool 2 — geocode the city  (real data: OSM Nominatim)
# ==========================================================================
def geocode(city: str) -> GeoLocation:
    """Resolve a city to coordinates. Raises osm.GeocodeError on failure."""
    return osm.geocode_city(city)


# ==========================================================================
# Tool 3 — weather for the upcoming Saturday  (real data: Open-Meteo)
# ==========================================================================
def fetch_weather(geo: GeoLocation) -> Weather | None:
    """Best-effort forecast. Returns None if the service is unavailable."""
    return weather_api.get_weather(geo.lat, geo.lon)


# ==========================================================================
# Tool 4 — find activity venues  (real data: OSM Overpass)
# ==========================================================================
def find_activities(profile: PreferenceProfile, geo: GeoLocation) -> list[POI]:
    """Search OpenStreetMap for things to do that match the user's interests.

    Interest-specific tags can be rare (e.g. "trekking" → peaks/hiking routes
    that OSM barely maps). When the interest search comes back thin, broaden it
    with generic outdoor + cultural venues so the planner has real choices
    instead of an empty itinerary.
    """
    pois = osm.search_pois(
        profile.activity_osm_filters, geo.bbox, category="activity",
        fallback_filters=osm.DEFAULT_ACTIVITY_FILTERS,
    )
    if len(pois) < 12:
        seen = {p.osm_id for p in pois}
        try:
            extra = osm.search_pois(
                osm.DEFAULT_ACTIVITY_FILTERS, geo.bbox, category="activity",
            )
            pois += [p for p in extra if p.osm_id not in seen]
        except httpx.HTTPError:
            pass  # keep whatever the first search returned
    return pois[:30]


# ==========================================================================
# Tool 5 — find food venues  (real data: OSM Overpass)
# ==========================================================================
def find_food(profile: PreferenceProfile, geo: GeoLocation) -> list[POI]:
    """Search OpenStreetMap for places to eat, then rank by constraint fit."""
    pois = osm.search_pois(
        profile.food_osm_filters, geo.bbox, category="food",
        fallback_filters=osm.DEFAULT_FOOD_FILTERS,
    )
    constraints = " ".join(profile.constraints).lower()
    wants_veg = "vegetarian" in constraints or "veg" in constraints
    wants_vegan = "vegan" in constraints

    if not (wants_veg or wants_vegan):
        return pois

    # Real signal: OSM tags diet:vegetarian / diet:vegan on many restaurants.
    def diet_score(poi: POI) -> int:
        tags = poi.tags
        score = 0
        if wants_vegan and tags.get("diet:vegan") in ("yes", "only"):
            score += 2
        if wants_veg and tags.get("diet:vegetarian") in ("yes", "only"):
            score += 2
        cuisine = str(tags.get("cuisine", "")).lower()
        if "vegetarian" in cuisine or "vegan" in cuisine:
            score += 1
        return score

    pois.sort(key=diet_score, reverse=True)
    return pois


# ==========================================================================
# Tool 6 — generate the itinerary draft  (LLM over real venue data)
# ==========================================================================
def _poi_line(index: int, poi: POI) -> str:
    extras = []
    for key in ("cuisine", "diet:vegetarian", "diet:vegan", "opening_hours", "fee"):
        if poi.tags.get(key):
            extras.append(f"{key}={poi.tags[key]}")
    extra = f" [{'; '.join(extras)}]" if extras else ""
    return f"  [{index}] {poi.name} — {poi.kind}{extra}"


def format_candidates(activities: list[POI], food: list[POI]) -> tuple[str, str, list[POI]]:
    """Render numbered candidate lists with a single shared index space.

    Returns (activities_text, food_text, combined_list) so a DraftItem.poi_index
    maps straight into `combined_list`.
    """
    combined = activities + food
    act_text = "\n".join(_poi_line(i, p) for i, p in enumerate(activities)) or "  (none found)"
    food_text = "\n".join(
        _poi_line(i + len(activities), p) for i, p in enumerate(food)
    ) or "  (none found)"
    return act_text, food_text, combined


def generate_draft(profile: PreferenceProfile, weather: Weather | None,
                    activities: list[POI], food: list[POI], llm: LLMClient,
                    *, violations: list[str] | None = None,
                    fallback: bool = False) -> tuple[DraftPlan, list[POI]]:
    """Ask the LLM for an itinerary draft. Returns (draft, candidate_list)."""
    act_text, food_text, combined = format_candidates(activities, food)

    if weather:
        weather_text = (
            f"{weather.description}, {weather.temp_min_c:.0f}-{weather.temp_max_c:.0f}°C, "
            f"{weather.precipitation_chance}% chance of rain. "
            f"{'Good for outdoor stops.' if weather.is_outdoor_friendly else 'Lean indoors.'}"
        )
    else:
        weather_text = "Forecast unavailable — assume mild, plan a sensible mix."

    replan_note = ""
    if violations:
        replan_note = prompts.REPLAN_NOTE_TEMPLATE.format(
            violations="\n".join(f"- {v}" for v in violations)
        ) + "\n\n"

    user_prompt = prompts.PLAN_USER_TEMPLATE.format(
        city=profile.city,
        currency_symbol=profile.currency_symbol,
        budget=profile.budget,
        currency=profile.currency,
        available_hours=profile.available_hours,
        start_preference=profile.start_preference,
        mood=profile.mood or "not specified",
        interests=", ".join(profile.interests) or "open to anything",
        constraints=", ".join(profile.constraints) or "none",
        weather=weather_text,
        activities=act_text,
        food=food_text,
        replan_note=replan_note,
    )
    system = prompts.FALLBACK_SYSTEM.format(city=profile.city) if fallback else prompts.PLAN_SYSTEM
    draft = llm.complete_json(user_prompt, DraftPlan, system=system, temperature=0.6)
    return draft, combined


def enrich_items(draft: DraftPlan, candidates: list[POI]) -> list[PlanItem]:
    """Attach real coordinates/OSM ids to the LLM's draft items.

    Anti-hallucination: when a stop references a real venue, its location_name
    is taken from OpenStreetMap data — never from the LLM's free text — so the
    place's identity always comes from the source of truth.
    """
    items: list[PlanItem] = []
    for di in sorted(draft.items, key=lambda d: d.order):
        poi = candidates[di.poi_index] if 0 <= di.poi_index < len(candidates) else None
        items.append(PlanItem(
            order=len(items) + 1,
            title=di.title,
            category=di.category,
            start_time=di.start_time,
            duration_minutes=max(0, di.duration_minutes),
            estimated_cost=max(0.0, di.estimated_cost),
            location_name=poi.name if poi else di.location_name,  # authoritative
            why_it_fits=di.why_it_fits,
            lat=poi.lat if poi else None,
            lon=poi.lon if poi else None,
            osm_id=poi.osm_id if poi else None,
        ))
    return items


def verify_grounding(items: list[PlanItem]) -> GroundingReport:
    """Anti-hallucination check — confirm venue stops trace back to real data.

    A 'venue stop' (food or activity) should be backed by a real OpenStreetMap
    place. Anything that isn't is flagged so the orchestrator and the UI can be
    honest about it.
    """
    venue_items = [i for i in items if i.category in ("food", "activity")]
    grounded = [i for i in venue_items if i.osm_id]
    ungrounded = [i.title for i in venue_items if not i.osm_id]
    score = len(grounded) / len(venue_items) if venue_items else 1.0
    return GroundingReport(
        venue_stops=len(venue_items),
        grounded_stops=len(grounded),
        grounding_score=round(score, 2),
        ungrounded=ungrounded,
    )


# ==========================================================================
# Tool 7 — estimate cost & time, including real travel between stops
# ==========================================================================
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def estimate_cost(items: list[PlanItem], profile: PreferenceProfile) -> CostBreakdown:
    """Sum venue spend and compute travel from the real coordinates of stops."""
    activities_cost = sum(i.estimated_cost for i in items if i.category != "food")
    food_cost = sum(i.estimated_cost for i in items if i.category == "food")

    total_km = 0.0
    last = None
    for item in items:
        if item.lat is not None and item.lon is not None:
            if last is not None:
                total_km += _haversine_km(last[0], last[1], item.lat, item.lon)
            last = (item.lat, item.lon)

    travel_minutes = round(total_km / _CITY_SPEED_KMH * 60)
    travel_cost = round(total_km * profile.transport_cost_per_km)
    stop_minutes = sum(i.duration_minutes for i in items)

    notes = []
    if total_km:
        notes.append(f"~{total_km:.1f} km of travel between stops "
                     f"(~{travel_minutes} min, {profile.currency_symbol}{travel_cost:.0f}).")
    notes.append("Venue costs are typical-spend estimates; OpenStreetMap has no price data.")

    return CostBreakdown(
        activities_cost=round(activities_cost),
        food_cost=round(food_cost),
        travel_cost=travel_cost,
        travel_minutes=travel_minutes,
        total_cost=round(activities_cost + food_cost + travel_cost),
        total_minutes=stop_minutes + travel_minutes,
        notes=notes,
    )


# ==========================================================================
# Tool 8 — validate the plan against budget, time and constraints
# ==========================================================================
def validate_plan(items: list[PlanItem], cost: CostBreakdown,
                   profile: PreferenceProfile, weather: Weather | None) -> ValidationResult:
    """Check the plan is affordable, fits the time window and is realistic."""
    violations: list[str] = []
    warnings: list[str] = []

    available_minutes = profile.available_hours * 60
    sym = profile.currency_symbol

    # Hard failures -> trigger a replan.
    if cost.total_cost > profile.budget:
        over = cost.total_cost - profile.budget
        violations.append(
            f"Plan costs {sym}{cost.total_cost:.0f}, which is {sym}{over:.0f} "
            f"over the {sym}{profile.budget:.0f} budget."
        )
    if cost.total_minutes > available_minutes + 20:  # 20-min grace
        over = cost.total_minutes - available_minutes
        violations.append(
            f"Plan needs ~{cost.total_minutes} min but only "
            f"{available_minutes:.0f} min are available ({over:.0f} min over)."
        )
    if not items:
        violations.append("The plan has no stops.")

    # Soft issues -> surfaced to the user, no replan.
    if cost.total_minutes < available_minutes * 0.5 and items:
        warnings.append(
            f"This plan only fills ~{cost.total_minutes} of "
            f"{available_minutes:.0f} available minutes — there's room for more."
        )
    if weather and not weather.is_outdoor_friendly:
        outdoor = [i for i in items if i.category == "activity" and i.osm_id]
        if outdoor:
            warnings.append(
                f"Weather looks {weather.description} — double-check the "
                "outdoor stops or keep a backup."
            )

    return ValidationResult(
        is_valid=not violations, violations=violations, warnings=warnings,
    )
