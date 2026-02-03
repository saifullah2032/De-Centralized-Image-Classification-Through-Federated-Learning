#!/bin/bash
# Start All Components Script for Linux/Mac
# This script starts the FL server, multiple clients, and the web interface

echo "================================"
echo "Starting Federated Learning System"
echo "================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down all components..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start FL Server
echo "Starting FL Server..."
python run_server.py --num-rounds 10 --min-clients 2 &
SERVER_PID=$!
sleep 5

# Start Clients
NUM_CLIENTS=5
echo "Starting $NUM_CLIENTS clients..."
for i in $(seq 0 $(($NUM_CLIENTS-1))); do
    echo "  Starting Client $i..."
    python run_client.py --client-id $i --num-clients $NUM_CLIENTS &
    sleep 2
done

# Start Web Interface
echo "Starting Web Interface..."
python run_web.py &
WEB_PID=$!

echo ""
echo "================================"
echo "All components started!"
echo "================================"
echo "FL Server:       Running (PID: $SERVER_PID)"
echo "Clients:         $NUM_CLIENTS running"
echo "Web Interface:   http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all components"
echo "================================"

# Wait for all background processes
wait
