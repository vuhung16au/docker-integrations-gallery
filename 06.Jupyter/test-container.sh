#!/bin/bash

echo "=== Jupyter Lab Container Test ==="
echo ""

echo "1. Checking container status..."
docker ps | grep jupyter-container

echo ""
echo "2. Checking container logs..."
docker logs jupyter-container | tail -10

echo ""
echo "3. Testing port connectivity..."
if curl -s http://localhost:8888 > /dev/null 2>&1; then
    echo "✓ Port 8888 is accessible"
else
    echo "✗ Port 8888 is not accessible"
fi

echo ""
echo "4. Container resource usage:"
docker stats jupyter-container --no-stream

echo ""
echo "=== Access Instructions ==="
echo "Open your browser and navigate to: http://localhost:8888/lab"
echo "**No authentication required!** Access Jupyter Lab directly without tokens or passwords."
echo ""
echo "To stop the container: docker stop jupyter-container"
echo "To remove the container: docker rm jupyter-container"
echo "To view logs: docker logs jupyter-container"
