#!/bin/bash

# Setup script for inference service

echo "Setting up inference service virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✓ Virtual environment created and dependencies installed!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the services:"
echo "  Terminal 1: python mock_metadata_service.py"
echo "  Terminal 2: python main.py"
echo "  Terminal 3: python mock_consumer.py"
echo ""
