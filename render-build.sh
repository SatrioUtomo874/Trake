#!/bin/bash
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating necessary directories and files..."
mkdir -p logs
touch trading_log.txt

echo "Build completed successfully!"
