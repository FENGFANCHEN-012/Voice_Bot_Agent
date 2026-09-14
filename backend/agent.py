from agent_service import agent_service


def run_agent(user_input: str, thread_id: str = "default") -> str:
    result = agent_service.run_agent(user_input, session_id=thread_id)
    return result["reply"]
