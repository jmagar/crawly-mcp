from fastmcp.prompts.prompt import Message

# Note: This file is intended to be imported into your main server.py
# from .prompts import example_prompt
#
# Then register it with your MCP instance:
# mcp.add_prompt(example_prompt)


def example_prompt(topic: str, tone: str = "professional") -> list[Message]:
    """This is an example prompt that generates a system and user message.

    Args:
        topic: The subject for the LLM to explain.
        tone: The desired tone for the explanation.

    Returns:
        A list of messages to guide the LLM.
    """
    system_message = Message(
        role="system",
        content=f"You are an expert in explaining complex topics in a {tone} manner.",
    )

    user_message = Message(
        role="user", content=f"Please explain the concept of {topic}."
    )

    return [system_message, user_message]
