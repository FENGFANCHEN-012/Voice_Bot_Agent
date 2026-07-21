import requests
from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool

BASE = "https://api-open.data.gov.sg/v2/real-time/api"
WIKI = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "SingaporeBot/1.0"


def _deg_to_dir(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


def _wiki_summary(title):
    resp = requests.get(f"{WIKI}/{title}", timeout=10, headers={"User-Agent": UA})
    if resp.status_code == 200:
        d = resp.json()
        return d.get("extract", "")
    return ""


def _wiki_search(query, limit=5):
    resp = requests.get(WIKI_API, params={
        "action": "query", "list": "search",
        "srsearch": query, "format": "json", "srlimit": limit,
    }, timeout=10, headers={"User-Agent": UA})
    return resp.json().get("query", {}).get("search", [])


# ─── Weather & Environment ───────────────────────────────────────

@tool
def get_weather(area: str) -> str:
    """Get the 2-hour weather forecast for a specific area in Singapore.

    Args:
        area: The name of the area in Singapore (e.g. 'Ang Mo Kio', 'Bedok', 'Clementi', 'Jurong East', 'Marina Bay', 'Sentosa')
    """
    resp = requests.get(f"{BASE}/two-hr-forecast", timeout=10)
    data = resp.json()
    forecasts = data["data"]["items"][0]["forecasts"]
    for item in forecasts:
        if item["area"].lower() == area.lower():
            return f"Weather in {area}: {item['forecast']}"
    all_areas = sorted(set(f["area"] for f in forecasts))
    return f"Area '{area}' not found. Available areas: {', '.join(all_areas)}"


@tool
def get_psi_reading() -> str:
    """Get the latest Pollutant Standards Index (PSI) readings for all regions in Singapore."""
    resp = requests.get(f"{BASE}/psi", timeout=10)
    data = resp.json()
    item = data["data"]["items"][0]
    readings = item["readings"]
    lines = ["Latest PSI readings in Singapore:"]
    for key, regions in readings.items():
        label = key.replace("_", " ").title()
        parts = [f"{r}: {v}" for r, v in regions.items()]
        lines.append(f"  {label}: {', '.join(parts)}")
    return "\n".join(lines)


@tool
def get_uv_index() -> str:
    """Get the latest UV index reading in Singapore."""
    resp = requests.get(f"{BASE}/uv", timeout=10)
    data = resp.json()
    records = data["data"]["records"]
    if not records or not records[0]["index"]:
        return "UV index data is currently unavailable."
    latest = records[0]["index"][0]
    value = latest["value"]
    hour = latest["hour"]
    if value <= 2:
        level = "Low"
    elif value <= 5:
        level = "Moderate"
    elif value <= 7:
        level = "High"
    elif value <= 10:
        level = "Very High"
    else:
        level = "Extreme"
    return f"UV Index at {hour}: {value} ({level}). {'Sun protection recommended.' if value > 2 else 'No sun protection needed.'}"


@tool
def get_air_temperature(area: str) -> str:
    """Get the current air temperature for a specific area in Singapore.

    Args:
        area: The name of the area in Singapore (e.g. 'Ang Mo Kio', 'Bedok', 'Jurong East', 'Marina Bay', 'Woodlands')
    """
    resp = requests.get(f"{BASE}/air-temperature", timeout=10)
    data = resp.json()
    stations = {s["id"]: s["name"] for s in data["data"]["stations"]}
    readings = data["data"]["readings"][0]["data"]
    station_value = {s["stationId"]: s["value"] for s in readings}
    for sid, sname in stations.items():
        if area.lower() in sname.lower():
            temp = station_value.get(sid)
            if temp is not None:
                return f"Current temperature at {sname}: {temp}°C"
            return f"Station '{sname}' found but no temperature data available."
    return (f"Area '{area}' not found. Sample stations: "
            + ", ".join(sorted(set(stations.values()))[:15]))


@tool
def get_wind_direction(area: str) -> str:
    """Get the current wind direction for a specific area in Singapore.

    Args:
        area: The name of the area in Singapore (e.g. 'Ang Mo Kio', 'Marina Bay', 'Woodlands', 'East Coast')
    """
    resp = requests.get(f"{BASE}/wind-direction", timeout=10)
    data = resp.json()
    stations = {s["id"]: s["name"] for s in data["data"]["stations"]}
    readings = data["data"]["readings"][0]["data"]
    for sid, sname in stations.items():
        if area.lower() in sname.lower():
            deg = next((r["value"] for r in readings if r["stationId"] == sid), None)
            if deg is not None:
                return f"Wind at {sname}: {deg}° ({_deg_to_dir(deg)})"
    return (f"Area '{area}' not found. Sample stations: "
            + ", ".join(sorted(set(stations.values()))[:10]))


@tool
def get_rainfall(area: str) -> str:
    """Get the current rainfall amount (mm) for a specific area in Singapore.

    Args:
        area: The name of the area in Singapore (e.g. 'Ang Mo Kio', 'Bukit Batok', 'Yio Chu Kang', 'East Coast')
    """
    resp = requests.get(f"{BASE}/rainfall", timeout=10)
    data = resp.json()
    stations = {s["id"]: s["name"] for s in data["data"]["stations"]}
    readings = data["data"]["readings"][0]["data"]
    for sid, sname in stations.items():
        if area.lower() in sname.lower():
            mm = next((r["value"] for r in readings if r["stationId"] == sid), None)
            if mm is not None:
                return f"Rainfall at {sname}: {mm} mm"
    return (f"Area '{area}' not found. Sample stations: "
            + ", ".join(sorted(set(stations.values()))[:10]))


@tool
def get_pm25() -> str:
    """Get the latest PM2.5 particulate matter readings by region in Singapore."""
    resp = requests.get(f"{BASE}/pm25", timeout=10)
    data = resp.json()
    readings = data["data"]["items"][0]["readings"]["pm25_one_hourly"]
    lines = ["Latest PM2.5 readings (μg/m³) by region:"]
    for region, value in readings.items():
        lines.append(f"  {region.title()}: {value}")
    return "\n".join(lines)


# ─── Transport ────────────────────────────────────────────────────

@tool
def get_mrt_info(query: str) -> str:
    """Get information about MRT/LRT lines and station connections in Singapore.

    Args:
        query: The MRT line name, station name, or question (e.g. 'North South Line', 'Jurong East station', 'which stations connect to Raffles Place')
    """
    search_q = f"{query} Singapore MRT"
    results = _wiki_search(search_q, limit=3)
    if not results:
        return f"No MRT information found for '{query}'."

    best = results[0]["title"]
    extract = _wiki_summary(best)
    if extract:
        return f"**{best}**\n\n{extract[:1500]}"
    return f"Found '{best}' but no details available."


@tool
def get_bus_info(query: str) -> str:
    """Get information about bus services and bus stops in Singapore.

    Args:
        query: The bus service number, bus interchange, or question (e.g. 'Bus 190', 'Ang Mo Kio bus interchange', 'Bus 36 route')
    """
    search_q = f"{query} Singapore bus"
    results = _wiki_search(search_q, limit=3)
    if not results:
        return f"No bus information found for '{query}'."

    best = results[0]["title"]
    extract = _wiki_summary(best)
    if extract:
        return f"**{best}**\n\n{extract[:1500]}"
    return f"Found '{best}' but no details available."


# ─── Time & Events ────────────────────────────────────────────────

@tool
def get_current_time() -> str:
    """Get the current date and time in Singapore (UTC+8)."""
    now = datetime.now(timezone(timedelta(hours=8)))
    return now.strftime("%A, %d %B %Y, %I:%M:%S %p (Singapore time, UTC+8)")


@tool
def get_singapore_events() -> str:
    """Get a list of notable upcoming or recent events, festivals, and holidays in Singapore."""
    pages = [
        "2026 in Singapore",
        "Public holidays in Singapore",
        "Singapore Grand Prix",
    ]
    seen = set()
    parts = []
    for page in pages:
        extract = _wiki_summary(page)
        if extract:
            key = extract[:100]
            if key not in seen:
                seen.add(key)
                parts.append(f"**{page}**\n{extract[:800]}")
    if parts:
        return "\n\n".join(parts)
    return "Could not retrieve event information at this time."


@tool
def lookup_attraction(query: str) -> str:
    """Look up a Singapore tourist attraction, resort, landmark or place on Wikipedia.

    Args:
        query: The name of the place, attraction, resort, or landmark (e.g. 'Marina Bay Sands', 'Sentosa', 'Gardens by the Bay')
    """
    search_q = f"{query} Singapore"
    results = _wiki_search(search_q, limit=3)
    if not results:
        return f"No Wikipedia article found for '{query}'."

    best = results[0]["title"]
    data = requests.get(f"{WIKI}/{best.replace(' ', '_')}", timeout=10,
                        headers={"User-Agent": UA}).json()
    if "extract" not in data:
        return f"Found '{best}' but no details available."

    desc = data.get("description", "")
    extract = data["extract"][:1200]
    result = f"**{data.get('title', best)}**"
    if desc:
        result += f"\n_{desc}_"
    result += f"\n\n{extract}"
    return result
