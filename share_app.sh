#!/bin/bash
# Script to launch Streamlit and create a public sharing link

echo "Starting StockSense AI Dashboard..."
source venv/bin/activate

# Check if streamlit is running on 8501
if ! lsof -i :8501 > /dev/null 2>&1; then
    streamlit run app/app.py --server.headless true --server.port 8501 &
    sleep 3
fi

echo "Creating public tunnel..."
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 nokey@localhost.run
