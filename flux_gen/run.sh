#!/bin/bash

# Step 1: Navigate to the ComfyUI directory and start the server in the background
cd $1 || { echo "Failed to enter ComfyUI directory"; exit 1; }
echo "Starting the server..."
python3 main.py &  # Start the server in the background
SERVER_PID=$!      # Capture the server process ID

# Step 2: Wait for a short duration to ensure the server is up
echo "Waiting for the server to start..."
sleep 15  # Adjust the sleep duration as necessary

# Step 3: Navigate to the test directory and start the application
cd $2 || { echo "Failed to enter test directory"; exit 1; }
echo "Starting the application..."
python3 ws_test.py $3

# Step 4: Clean up
echo "Shutting down the server..."
kill $SERVER_PID  # Terminate the server process when the application exits
