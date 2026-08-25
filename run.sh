#!/bin/bash

hf download cross-encoder/ms-marco-MiniLM-L-6-v2
export HF_HUB_OFFLINE=1

# init venv
source .venv/bin/activate
echo "venv activated"

# Script to run FastAPI backend and Streamlit frontend simultaneously
# Define the commands
FASTAPI_CMD="uvicorn app.main:app --host=0.0.0.0 --port=8000 --reload"  # Replace with your FastAPI file path
STREAMLIT_CMD="streamlit run web_ui.py --server.address=0.0.0.0"  # Replace with your Streamlit file path

# Function to kill both processes when script exits
cleanup() {
    echo "Stopping processes..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

cd backend
# Start FastAPI backend
echo "Starting FastAPI backend..."
eval $FASTAPI_CMD &
FASTAPI_PID=$!

cd ../frontend
# Start Streamlit frontend
echo "Starting Streamlit frontend..."
eval $STREAMLIT_CMD &
STREAMLIT_PID=$!

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID