from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import settings
from guardrails import guardrail_check, infer_intent, sanitize_user_input
from session_store import session_store
from tools import (
    get_air_temperature,
    get_bus_info,
    get_current_time,
    get_mrt_info,
    get_pm25,
    get_psi_reading,
    get_rainfall,
    get_singapore_events,
    get_uv_index,
    get_weather,
    get_wind_direction,
    lookup_attraction,
)

SYSTEM_PROMPT = (
    "You are a professional Singapore travel and information assistant. "
    "Give concise but helpful answers. Use tool outputs when they are relevant and present them naturally. "
    "If information is missing, ask a brief clarifying question. Keep the tone professional, warm, and clear."
)


class AgentService:
    def __init__(self) -> None:
        self.model = ChatOllama(model=settings.model_name, temperature=settings.temperature)
        self.tools = [
            get_weather,
            get_psi_reading,
            get_uv_index,
            get_air_temperature,
            get_wind_direction,
            get_rainfall,
            get_pm25,
            lookup_attraction,
            get_mrt_info,
            get_bus_info,
            get_current_time,
            get_singapore_events,
        ]
        self.agent = create_react_agent(
            self.model,
            self.tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
        )

    def _prepare_history(self, session_id: str, limit: int | None = None) -> list[tuple[str, str]]:
        history = session_store.get_history(session_id, limit=limit or settings.max_history)
        return [(entry["role"], entry["content"]) for entry in history if entry.get("role") in {"user", "assistant"}]

    def _run_langgraph(self, user_input: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}
        result = self.agent.invoke({"messages": [("user", user_input)]}, config)
        return result["messages"][-1].content

    def run_agent(self, user_input: str, session_id: str = "default") -> dict[str, Any]:
        cleaned = sanitize_user_input(user_input)
        guard = guardrail_check(cleaned)
        if not guard["safe"]:
            session_store.append_turn(session_id, "assistant", guard["message"], {"guardrail": guard["reason"]})
            return {
                "reply": guard["message"],
                "session_id": session_id,
                "intent": "guardrail",
                "safe": False,
            }

        intent = infer_intent(cleaned)
        session_store.append_turn(session_id, "user", cleaned, {"intent": intent})

        try:
            reply = self._run_langgraph(cleaned, session_id)
        except Exception as exc:
            fallback = "I am having trouble processing that right now. Please try again in a moment."
            session_store.append_turn(session_id, "assistant", fallback, {"error": str(exc)})
            return {
                "reply": fallback,
                "session_id": session_id,
                "intent": intent,
                "safe": True,
            }

        session_store.append_turn(session_id, "assistant", reply, {"intent": intent})
        return {
            "reply": reply,
            "session_id": session_id,
            "intent": intent,
            "safe": True,
        }


agent_service = AgentService()
