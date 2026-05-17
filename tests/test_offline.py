"""Offline tests — no network, no API key required.

Run:  .venv/bin/python -m pytest tests/ -q
(or just .venv/bin/python tests/test_offline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.models import CostBreakdown, PlanItem, PreferenceProfile
from agent.tools import _haversine_km, estimate_cost, validate_plan
from data import osm, weather


def _item(order, cat, mins, cost, lat=None, lon=None):
    return PlanItem(order=order, title=f"Stop {order}", category=cat, start_time="10:00",
                    duration_minutes=mins, estimated_cost=cost, why_it_fits="fits",
                    lat=lat, lon=lon)


def test_haversine_known_distance():
    # Bangalore centre -> Indiranagar, ~5 km.
    km = _haversine_km(12.9716, 77.5946, 12.9783, 77.6408)
    assert 4.0 < km < 6.5, km


def test_next_saturday_is_a_saturday():
    assert weather.next_saturday().weekday() == 5


def test_filter_sanitizer_drops_garbage():
    good = ['["amenity"="cafe"]']
    bad = ['DROP TABLE', 'amenity=cafe', '["x"="y"]; evil']
    assert osm._sanitize_filters(good + bad, osm.DEFAULT_FOOD_FILTERS) == good
    # Empty input -> falls back to defaults.
    assert osm._sanitize_filters([], osm.DEFAULT_FOOD_FILTERS) == osm.DEFAULT_FOOD_FILTERS


def test_estimate_cost_sums_and_adds_travel():
    items = [_item(1, "activity", 90, 0, 12.97, 77.59),
             _item(2, "food", 60, 400, 12.98, 77.64)]
    profile = PreferenceProfile(city="Bangalore", budget=2000, available_hours=4)
    cost = estimate_cost(items, profile)
    assert cost.food_cost == 400
    assert cost.activities_cost == 0
    assert cost.travel_cost > 0          # real distance between the two stops
    assert cost.total_minutes > 150      # 150 min of stops + travel


def test_validate_flags_over_budget():
    items = [_item(1, "food", 60, 5000)]
    profile = PreferenceProfile(city="Bangalore", budget=500, available_hours=4)
    result = validate_plan(items, estimate_cost(items, profile), profile, None)
    assert not result.is_valid
    assert any("budget" in v for v in result.violations)


def test_validate_flags_over_time():
    items = [_item(1, "activity", 600, 0)]  # 10 hours
    profile = PreferenceProfile(city="Bangalore", budget=99999, available_hours=4)
    result = validate_plan(items, estimate_cost(items, profile), profile, None)
    assert not result.is_valid
    assert any("min over" in v or "available" in v for v in result.violations)


def test_validate_passes_reasonable_plan():
    items = [_item(1, "activity", 90, 0, 12.97, 77.59),
             _item(2, "food", 60, 400, 12.98, 77.60)]
    profile = PreferenceProfile(city="Bangalore", budget=2000, available_hours=4)
    result = validate_plan(items, estimate_cost(items, profile), profile, None)
    assert result.is_valid


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} offline tests passed.")
