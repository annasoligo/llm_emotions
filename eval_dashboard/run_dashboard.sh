#!/bin/bash
# Convenience script to run the Emotion Onset Dashboard

cd /workspace-vast/annas/git/research-tools

# Activate virtual environment if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Run Streamlit
streamlit run eval_dashboard/app.py \
    --server.port 8501 \
    --server.address localhost \
    --theme.base light

# Alternative: Run on specific port with custom settings
# streamlit run eval_dashboard/app.py --server.port 8502 --theme.base dark
