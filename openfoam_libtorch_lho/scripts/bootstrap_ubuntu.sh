#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for Ubuntu system packages and OpenFOAM v14 installation
# Idempotent: safe to run multiple times

echo "=== Installing system packages ==="

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    software-properties-common \
    pkg-config \
    python3 \
    python3-venv \
    python3-pip \
    ninja-build

echo "=== Checking OpenFOAM v14 installation ==="

# Check if OpenFOAM v14 is already installed
if command -v foamVersion &> /dev/null && [[ "$(foamVersion 2>/dev/null)" == *"14"* ]]; then
    echo "OpenFOAM v14 appears to be already installed and available."
    foamVersion
else
    echo "Installing OpenFOAM Foundation v14..."

    sudo add-apt-repository -y universe

    sudo sh -c \
      "wget -O - https://dl.openfoam.org/gpg.key \
       > /etc/apt/trusted.gpg.d/openfoam.asc"

    sudo rm -f /etc/apt/sources.list.d/*dl_openfoam_org*list 2>/dev/null || true

    sudo add-apt-repository -y \
      "http://dl.openfoam.org/ubuntu main dev"

    # For Ubuntu 26.04, check for amd64v3 warning and fix if needed
    UBUNTU_CODENAME=$(grep "^VERSION_CODENAME=" /etc/os-release | cut -d= -f2)
    if [[ "$UBUNTU_CODENAME" == "whiskers" ]]; then
        echo "Ubuntu 26.04 detected - checking for amd64v3 architecture requirement..."
        # The sed will adjust the source line if needed
        sudo sed -i \
          "s/^\(deb\).*\(http\)/\1 [arch=amd64] \2/" \
          /etc/apt/sources.list.d/*dl_openfoam_org*list 2>/dev/null || true
    fi

    sudo apt update
    sudo apt install -y openfoam14
fi

echo "=== Testing OpenFOAM installation in clean shell ==="

# Test in a clean shell with explicit sourcing
source /opt/openfoam14/etc/bashrc
foamVersion
foamRun -help | head -5
echo "WM_PROJECT_VERSION: $WM_PROJECT_VERSION"
echo "WM_OPTIONS: $WM_OPTIONS"
echo "FOAM_USER_APPBIN: $FOAM_USER_APPBIN"

echo "=== Creating user directories ==="

mkdir -p "$FOAM_USER_APPBIN"
mkdir -p "$WM_PROJECT_USER_DIR/applications"

echo "=== Bootstrap complete ==="
echo "To use OpenFOAM v14, run:"
echo "  source /opt/openfoam14/etc/bashrc"
