"""Open-Meteo client — real forecast for the upcoming Saturday.

Free, keyless, 16-day forecast range. Used to bias the plan toward indoor or
outdoor stops depending on actual conditions.
"""

from __future__ import annotations

from datetime import date, timedelta
import httpx

from agent.models import Weather

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> human description.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def next_saturday(today: date | None = None) -> date:
    """Return the date of the coming Saturday (today if today is Saturday)."""
    today = today or date.today()
    return today + timedelta(days=(5 - today.weekday()) % 7)


def get_weather(lat: float, lon: float, *, timeout: float = 15.0) -> Weather | None:
    """Fetch the forecast for the next Saturday.

    Returns None on any failure — weather is a nice-to-have, not a hard
    dependency, so the agent degrades gracefully without it.
    """
    target = next_saturday()
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max",
        "timezone": "auto",
        "start_date": target.isoformat(),
        "end_date": target.isoformat(),
    }
    try:
        resp = httpx.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        code = int(daily["weather_code"][0])
        tmax = float(daily["temperature_2m_max"][0])
        tmin = float(daily["temperature_2m_min"][0])
        rain = int(daily["precipitation_probability_max"][0] or 0)
    except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError):
        return None

    description = _WMO.get(code, "mixed conditions")
    outdoor_ok = rain < 50 and code < 61 and 8 <= tmax <= 38

    return Weather(
        date=target.isoformat(),
        description=description,
        temp_max_c=tmax,
        temp_min_c=tmin,
        precipitation_chance=rain,
        is_outdoor_friendly=outdoor_ok,
    )
