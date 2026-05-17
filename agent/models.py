"""Typed data models shared across the agent.

Everything the agent reasons about is a Pydantic model so we get validation,
JSON (de)serialisation and a stable contract between tools for free.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Preferences
# --------------------------------------------------------------------------
class PreferenceProfile(BaseModel):
    """The user's request, normalised into a structured profile by the LLM.

    Accepts either a filled form or a free-text paragraph as raw input.
    """

    city: str = Field(description="City the user wants to spend Saturday in.")
    budget: float = Field(description="Total budget as a number, in local currency.")
    currency: str = Field(default="INR", description="ISO currency code, inferred from the city.")
    currency_symbol: str = Field(default="₹", description="Currency symbol for display.")
    available_hours: float = Field(default=4.0, description="Hours the user has free.")
    start_preference: str = Field(
        default="flexible",
        description="When they'd like to start: morning / afternoon / evening / flexible.",
    )
    transport_cost_per_km: float = Field(
        default=15.0,
        description="Typical local ride-hail/auto cost per km, in local currency.",
    )
    mood: str = Field(default="", description="The user's mood / energy level in their words.")
    interests: list[str] = Field(default_factory=list, description="Things they enjoy.")
    constraints: list[str] = Field(default_factory=list, description="Hard limits / things to avoid.")

    # Overpass (OpenStreetMap) tag selectors the LLM derived from the interests.
    # Keeps venue discovery fully dynamic — no fixed category list.
    activity_osm_filters: list[str] = Field(default_factory=list)
    food_osm_filters: list[str] = Field(default_factory=list)

    # Clarifying-question flow.
    needs_clarification: bool = Field(default=False)
    clarifying_questions: list[str] = Field(default_factory=list)

    normalized_summary: str = Field(
        default="", description="One-line restatement of what the user asked for."
    )


# --------------------------------------------------------------------------
# Places (real data from OpenStreetMap)
# --------------------------------------------------------------------------
class GeoLocation(BaseModel):
    """Resolved coordinates + search box for a city."""

    city: str
    display_name: str
    lat: float
    lon: float
    # Bounding box clamped to a city-scale search area: south, west, north, east.
    bbox: tuple[float, float, float, float]


class Weather(BaseModel):
    """Forecast for the upcoming Saturday."""

    date: str
    description: str
    temp_max_c: float
    temp_min_c: float
    precipitation_chance: int = Field(description="Percent chance of rain.")
    is_outdoor_friendly: bool


class POI(BaseModel):
    """A real point of interest pulled from OpenStreetMap."""

    osm_id: str
    name: str
    category: str = Field(description="Coarse bucket: activity or food.")
    kind: str = Field(description="Specific type, e.g. cafe, park, museum.")
    lat: float
    lon: float
    tags: dict[str, Any] = Field(default_factory=dict)

    def maps_url(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_id}"


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------
class PlanItem(BaseModel):
    """One stop in the Saturday itinerary."""

    order: int
    title: str
    category: str = Field(description="food / activity / break / travel")
    start_time: str = Field(description="HH:MM, 24-hour.")
    duration_minutes: int
    estimated_cost: float
    location_name: str = ""
    why_it_fits: str = Field(description="Why this stop suits the user.")
    lat: Optional[float] = None
    lon: Optional[float] = None
    osm_id: Optional[str] = None


class CostBreakdown(BaseModel):
    """Output of the estimate_cost tool."""

    activities_cost: float = 0.0
    food_cost: float = 0.0
    travel_cost: float = 0.0
    travel_minutes: int = 0
    total_cost: float = 0.0
    total_minutes: int = 0
    notes: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Output of the validate_plan tool."""

    is_valid: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """The final itinerary handed back to the UI."""

    items: list[PlanItem]
    summary: str
    tradeoffs: list[str] = Field(default_factory=list)
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    validation: ValidationResult = Field(
        default_factory=lambda: ValidationResult(is_valid=True)
    )
    weather_note: str = ""
    is_fallback: bool = False


# --------------------------------------------------------------------------
# LLM draft output (raw plan before enrichment + costing)
# --------------------------------------------------------------------------
class DraftItem(BaseModel):
    """One stop as proposed by the planning LLM, referencing a candidate venue."""

    order: int
    title: str
    category: str = Field(description="food / activity / break / travel")
    start_time: str
    duration_minutes: int
    estimated_cost: float
    poi_index: int = Field(description="Index into the candidate list, or -1 if none.")
    location_name: str = ""
    why_it_fits: str


class DraftPlan(BaseModel):
    """The planning LLM's full proposal, before code-side costing/validation."""

    items: list[DraftItem]
    summary: str
    tradeoffs: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Agent trace
# --------------------------------------------------------------------------
class StepStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class EventKind(str, Enum):
    STEP = "step"          # a tool started / finished
    THINKING = "thinking"  # a short reasoning note
    CLARIFY = "clarify"    # agent needs the user to answer questions
    ERROR = "error"        # a hard failure surfaced to the user
    RESULT = "result"      # the final plan is ready


class TraceEvent(BaseModel):
    """A single streamed update from the orchestrator."""

    kind: EventKind
    tool: str = ""
    status: StepStatus = StepStatus.RUNNING
    message: str = ""
    detail: Optional[str] = None
    elapsed_ms: Optional[int] = None

    # Populated only on CLARIFY / RESULT events.
    questions: list[str] = Field(default_factory=list)
    profile: Optional[PreferenceProfile] = None
    plan: Optional[Plan] = None
