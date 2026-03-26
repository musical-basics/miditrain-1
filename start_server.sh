#!/bin/bash

# Configuration
PORT=8000

# 1. Self-Healing: Check if the port is already in use
echo "🕵️ Checking for any rogue processes running on port $PORT..."
PID=$(lsof -t -i:$PORT)

if [ -n "$PID" ]; then
    echo "⚠️ Found process $PID hogging port $PORT. Terminating it to clear the way..."
    kill -9 $PID
    # Wait a brief moment to ensure the OS frees the port completely
    sleep 1
    echo "✅ Port $PORT successfully cleared!"
else
    echo "✅ Port $PORT is already free."
fi

# 2. Start the Application
echo "🚀 Activating virtual environment and booting the MidiCorrector server..."
source .venv/bin/activate
uvicorn api:app --port $PORT
