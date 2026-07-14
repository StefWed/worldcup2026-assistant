"""
Tool definitions and implementations for the agent:
- search_document: RAG search over the team-description PDF
- search_web: DuckDuckGo web search
- analyze_match: simple stats-based match prediction
"""

from duckduckgo_search import DDGS

from config import TEAM_STATS
from rag import search as rag_search

# --- Module-level state for the RAG index ---
# Set once via init_document_search() after the pipeline has been built
# (e.g. in app.py, on PDF upload). Tool functions read from here so their
# signatures only need the args the LLM is meant to supply.
_INDEX = None
_DOCUMENTS = None


def init_document_search(index, documents):
    """Call this once after run_pipeline() to make search_document usable."""
    global _INDEX, _DOCUMENTS
    _INDEX = index
    _DOCUMENTS = documents


def search_document(query: str) -> str:
    """Wraps rag.search(). Requires init_document_search() to have been called."""
    if _INDEX is None or _DOCUMENTS is None:
        return "Document index not initialized yet — please upload/build the PDF index first."

    results = rag_search(query, _INDEX, _DOCUMENTS, k=3)
    parts = []
    for i, doc in enumerate(results, 1):
        parts.append(f"[Source {i} | Team: {doc['team']}]\n{doc['text']}")
    return "\n\n".join(parts)


def search_web(query: str) -> str:
    """DuckDuckGo search."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        parts = []
        for r in results:
            parts.append(f"{r['title']}\n{r['body']}\nSource: {r['href']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Web search failed: {e}"


def get_team_stats(team):
    return TEAM_STATS.get(team, None)


def analyze_match(team1: str, team2: str) -> str:
    stats1 = get_team_stats(team1)
    stats2 = get_team_stats(team2)
    if not stats1 or not stats2:
        return "One or both teams not found."

    score1 = stats1["win_rate"] + (stats1["avg_goals"] - stats1["goals_conceded"])
    score2 = stats2["win_rate"] + (stats2["avg_goals"] - stats2["goals_conceded"])

    if score1 > score2:
        winner = team1
    elif score2 > score1:
        winner = team2
    else:
        winner = "Draw"

    return (
        f"{team1} score: {score1:.2f}\n"
        f"{team2} score: {score2:.2f}\n"
        f"Predicted winner: {winner}"
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Search the PDF for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for external information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_match",
            "description": (
                "Analyze a football match using team statistics. "
                "Use this when asked to predict a match outcome."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team1": {"type": "string"},
                    "team2": {"type": "string"}
                },
                "required": ["team1", "team2"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "search_document": search_document,
    "search_web": search_web,
    "analyze_match": analyze_match,
}
