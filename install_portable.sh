#!/bin/bash
# Installation script for UserRL environment on a new machine
# This script handles the installation of all dependencies including local gym packages

set -e  # Exit on error

echo "================================"
echo "UserRL Environment Setup"
echo "================================"

# Check if we're in the correct directory
if [ ! -f "setup.py" ]; then
    echo "Error: This script must be run from the UserRL root directory"
    exit 1
fi

# Step 1: Install main dependencies
echo ""
echo "Step 1: Installing main dependencies from requirements_portable.txt..."
pip install -r requirements_portable.txt

# Step 2: Install flash-attn separately (optional, but recommended)
echo ""
echo "Step 2: Installing flash-attn..."
echo "Note: This may take several minutes and requires CUDA to be available"
read -p "Do you want to install flash-attn? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install flash-attn --no-build-isolation
else
    echo "Skipping flash-attn installation"
fi

# Step 3: Install local gym packages
echo ""
echo "Step 3: Installing local gym packages..."
GYMS=(
    "AlfworldGym"
    "FunctionGym"
    "IntentionGym"
    "InteractCompGym"
    "PersuadeGym"
    "SearchGym"
    "TauGym"
    "TelepathyGym"
    "TravelGym"
    "TurtleGym"
)

for gym in "${GYMS[@]}"; do
    if [ -d "gyms/$gym" ]; then
        echo "Installing $gym..."
        pip install -e "./gyms/$gym"
    else
        echo "Warning: gyms/$gym not found, skipping..."
    fi
done

echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "Summary:"
echo "- Main dependencies installed from requirements_portable.txt"
echo "- Flash-attn: $(if [[ $REPLY =~ ^[Yy]$ ]]; then echo "Installed"; else echo "Skipped"; fi)"
echo "- Local gym packages installed in editable mode"
echo ""
echo "Note: Make sure your CUDA environment is properly configured."
echo "You can verify the installation by running: python -c 'import torch; print(torch.cuda.is_available())'"
