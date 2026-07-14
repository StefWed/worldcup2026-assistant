# World Cup Assistent

A small Streamlit app wrapping the RAG + tool-calling agent: search team
descriptions from a PDF, search the web, or run a simple stats-based match
prediction.

## Files

- `config.py` — OpenAI client setup, model names, team stats
- `rag.py` — PDF loading, cleaning, embeddings, FAISS vector search
- `tools.py` — the three agent tools (`search_document`, `search_web`, `analyze_match`) and their schemas
- `agent.py` — the tool-calling loop
- `app.py` — the Streamlit UI

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key either as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

or in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

## Run

```bash
streamlit run app.py
```

Then, in the sidebar, upload your team-description PDF (one team per page,
team name as the first line of each page) and click **Build index**. Once
built, ask questions in the main panel — the agent decides whether to pull
from the PDF, search the web, or run the match-prediction tool, and the
"Show tool activity" expander under each answer shows exactly which tool(s)
were called and what they returned.


## About `WorldCupTeamProfiles.pdf`

This file is included in the repo as the source/reference copy of the team
descriptions used in the demo. It is **not** loaded automatically by the app.

To use it: run the app, then in the sidebar upload `WorldCupTeamProfiles.pdf`
(select it from the repo/Codespace file browser) and click **Build index**.
The app builds the vector index fresh from whatever PDF you upload each
session — so you can also swap in a different team-profile PDF without
touching any code, as long as it follows the same format (one team per page,
team name as the first line of each page).


## About `team_stats.csv`

The statistics in this file (`win_rate`, `avg_goals`, `goals_conceded`) are
**synthesized/illustrative values**, not real tournament data. They exist to
demonstrate the `analyze_match` tool's simple scoring model, not to make an
accurate prediction claim. If you want real predictive value, replace this
file with actual team statistics from a reliable source (e.g. FIFA rankings,
historical qualifying results, or a proper analytics provider).