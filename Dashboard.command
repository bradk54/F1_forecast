#!/bin/bash
#
# Double-click this in Finder to open the dashboard.
#
# It lives at the repository root and resolves its own directory, so it works
# from Finder (which starts you in $HOME) as well as from a terminal.  The
# server runs in the foreground: this window *is* the server, and closing it
# stops the app.
#
# Also runnable as ./Dashboard.command, or from anywhere by full path.

set -u

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
PORT=8501
URL="http://localhost:${PORT}"

printf '\033]0;F1 dashboard\007'   # name the Terminal tab
echo "F1 retirement dashboard"
echo "$ROOT"
echo

# Already up?  Reopening the browser is the whole job -- starting a second
# server would fail on the port and look like a broken button.
if lsof -ti :"$PORT" >/dev/null 2>&1; then
  echo "Already running on port $PORT. Opening $URL"
  open "$URL"
  echo
  echo "This window is not the server -- the original one is."
  echo "Press any key to close."
  read -r -n 1 -s
  exit 0
fi

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: no interpreter at .venv/bin/python"
  echo
  echo "Create it, then install the requirements:"
  echo "    python3 -m venv .venv"
  echo "    ./.venv/bin/pip install -r requirements.txt"
  echo
  echo "Press any key to close."
  read -r -n 1 -s
  exit 1
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "ERROR: streamlit is not installed in .venv"
  echo
  echo "    ./.venv/bin/pip install -r requirements.txt"
  echo
  echo "Press any key to close."
  read -r -n 1 -s
  exit 1
fi

if [ ! -f "$ROOT/Data/processed/dnf_dataset.parquet" ]; then
  # Not fatal: the app has a page that explains this and gives the command.
  echo "NOTE: no dataset yet at Data/processed/dnf_dataset.parquet."
  echo "      The app will open and tell you how to build one."
  echo
fi

# Open the browser shortly after the server starts.  Streamlit's own
# --server.headless=false races the bind and sometimes opens a dead tab.
( sleep 2; open "$URL" ) &

echo "Starting on $URL"
echo "Close this window to stop the server."
echo

exec "$PY" -m streamlit run app/Home.py \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
