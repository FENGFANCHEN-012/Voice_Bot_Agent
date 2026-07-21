from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import (
    get_weather, get_psi_reading, get_uv_index, get_air_temperature,
    get_wind_direction, get_rainfall, get_pm25, lookup_attraction,
    get_mrt_info, get_bus_info, get_current_time, get_singapore_events,
)

SYSTEM_PROMPT = (
    "You are a helpful Singapore tourism assistant. "
    "Answer clearly and concisely about attractions, transportation, food, "
    "local customs, and activities in Singapore. "
    "Always keep a friendly and informative tone. "
    "If you are unsure about something, say so honestly.\n\n"
    "You have access to real-time data tools. Use them when the user asks:\n"
    "- Weather forecast for an area → get_weather\n"
    "- Air quality / PSI → get_psi_reading\n"
    "- UV index / sun protection → get_uv_index\n"
    "- Temperature in an area → get_air_temperature\n"
    "- Wind direction / speed in an area → get_wind_direction\n"
    "- Rainfall in an area → get_rainfall\n"
    "- PM2.5 air pollution → get_pm25\n"
    "- Info about an attraction, resort, or event → lookup_attraction\n"
    "- MRT / LRT line and station info → get_mrt_info\n"
    "- Bus services, routes, interchanges → get_bus_info\n"
    "- Current date and time in Singapore → get_current_time\n"
    "- Upcoming events, festivals, holidays in Singapore → get_singapore_events\n\n"
    "When you get tool results, present them naturally to the user in a friendly way."
)

model = ChatOllama(model="minimax-m3:cloud", temperature=0.7)
tools = [
    get_weather, get_psi_reading, get_uv_index, get_air_temperature,
    get_wind_direction, get_rainfall, get_pm25, lookup_attraction,
    get_mrt_info, get_bus_info, get_current_time, get_singapore_events,
]

agent = create_react_agent(
    model,
    tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=MemorySaver(),
)


def run_agent(user_input: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [("user", user_input)]},
        config,
    )
    return result["messages"][-1].content
