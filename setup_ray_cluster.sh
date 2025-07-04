#!/bin/bash

echo "=== LocalTranslate Ray Cluster Setup ==="
echo

# Function to get local IP address
get_local_ip() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}'
    else
        # Linux
        hostname -I | awk '{print $1}'
    fi
}

if [ "$1" == "head" ]; then
    echo "Setting up Ray HEAD node (Mac Mini)..."
    echo
    
    # Get local IP
    LOCAL_IP=$(get_local_ip)
    echo "Your Mac Mini's IP address is: $LOCAL_IP"
    echo
    echo "Starting Ray head node..."
    ray stop --force 2>/dev/null
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    
    echo
    echo "Ray head node started!"
    echo
    echo "=== IMPORTANT: Save this information for your Linux machine ==="
    echo "Ray cluster address: $LOCAL_IP:6379"
    echo "================================================================"
    echo
    echo "Now you can start the FastAPI server with:"
    echo "export HF_TOKEN=your_token_here"
    echo "python3 -m uvicorn app.main:app --reload --host 0.0.0.0"
    
elif [ "$1" == "worker" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide the head node address"
        echo "Usage: ./setup_ray_cluster.sh worker HEAD_IP:6379"
        exit 1
    fi
    
    echo "Setting up Ray WORKER node (Linux with GPU)..."
    echo "Connecting to head node at: $2"
    echo
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        echo "Found $GPU_COUNT GPU(s)"
        ray stop --force 2>/dev/null
        ray start --address="$2" --num-gpus=$GPU_COUNT
    else
        echo "No GPU detected, starting CPU-only worker"
        ray stop --force 2>/dev/null
        ray start --address="$2"
    fi
    
    echo
    echo "Ray worker node connected!"
    
elif [ "$1" == "stop" ]; then
    echo "Stopping Ray on this machine..."
    ray stop --force
    echo "Ray stopped."
    
else
    echo "LocalTranslate Ray Cluster Setup"
    echo
    echo "Usage:"
    echo "  On Mac Mini (head):     ./setup_ray_cluster.sh head"
    echo "  On Linux (worker):      ./setup_ray_cluster.sh worker MAC_MINI_IP:6379"
    echo "  To stop Ray:            ./setup_ray_cluster.sh stop"
    echo
    echo "Example:"
    echo "  Mac Mini:   ./setup_ray_cluster.sh head"
    echo "  Linux:      ./setup_ray_cluster.sh worker 192.168.1.100:6379"
fi 