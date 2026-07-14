"""
The agent loop: gives the LLM access to the three tools and lets it decide
which to call, across up to MAX_STEPS rounds, before returning a final answer.
"""

import json

from config import client, AGENT_MODEL
from tools import TOOLS, TOOL_FUNCTIONS

MAX_STEPS = 5

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "You have access to three tools:\n"
    "- search_document: for information from the uploaded PDF\n"
    "- search_web: for general or current information\n"
    "- analyze_match: for predicting match outcomes using team statistics\n\n"
    "Guidelines:\n"
    "- Use search_document for questions about team playing styles or descriptions\n"
    "- Use search_web for general knowledge or recent events\n"
    "- Use analyze_match when asked to predict or compare match outcomes\n\n"
    "Always base your final answer on the tool results."
)


def run_agent(user_question: str):
    """
    Runs the tool-calling loop for a single question.

    Returns a dict:
        {
            "answer": str,
            "trace": [ {"tool": str, "args": dict, "result": str}, ... ]
        }
    so the Streamlit UI can show both the final answer and the tool activity,
    instead of relying on printed output.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    trace = []

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                result = TOOL_FUNCTIONS[fn_name](**fn_args)

                trace.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result": result
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            return {"answer": message.content, "trace": trace}

    return {
        "answer": "Agent exceeded maximum steps without a final answer.",
        "trace": trace
    }
