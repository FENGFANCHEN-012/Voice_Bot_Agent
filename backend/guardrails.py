import re
from typing import Any


def sanitize_user_input(text: str) -> str:
    if text is None:
        return ""
    clean = text.strip()
    clean = re.sub(r"\s+", " ", clean)
    clean = clean[:4000]
    return clean


def guardrail_check(text: str) -> dict[str, Any]:
    cleaned = sanitize_user_input(text)
    if not cleaned:
        return {"safe": False, "reason": "empty_input", "message": "I didn't catch that. Could you say it again?"}
    if len(cleaned) > 4000:
        return {"safe": False, "reason": "input_too_long", "message": "Your message is too long. Please shorten it and try again."}
    return {"safe": True, "reason": None, "message": cleaned}


def infer_intent(user_input: str) -> str:
    text = user_input.lower()
    if any(keyword in text for keyword in ["weather", "rain", "sunny", "cloudy", "temperature", "humid", "forecast"]):
        return "weather"
    if any(keyword in text for keyword in ["mrt", "bus", "train", "station", "route", "transport", "commute"]):
        return "transport"
    if any(keyword in text for keyword in ["attraction", "place", "museum", "park", "sentosa", "marina", "tour", "visit"]):
        return "attraction"
    if any(keyword in text for keyword in ["time", "date", "today", "now", "clock"]):
        return "time"
    if any(keyword in text for keyword in ["event", "festival", "holiday", "happening"]):
        return "event"
    return "general"
