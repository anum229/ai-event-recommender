#!/bin/bash

echo "🔄 Running extract_data.py..."
python extract_data.py

echo "📊 Running vectorize_data.py..."
python vectorize_data.py

echo "🚀 Starting API server..."
python api_server.py