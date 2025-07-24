#!/usr/bin/env python3
"""
Script to install TensorFlow with GPU support for CUDA 12.x
"""

import subprocess
import sys
import os

def main():
    """Install TensorFlow with GPU support."""
    print("Installing TensorFlow with GPU support...")
    
    # Activate virtual environment if not already activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("Please run this script after activating your virtual environment.")
        return 1
    
    # Uninstall existing TensorFlow installations
    print("Removing existing TensorFlow installations...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-gpu', 'tensorflow-intel'])
    
    # Install specific version of TensorFlow compatible with CUDA 12.x
    print("Installing TensorFlow 2.10.0 with GPU support...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.10.0'])
    
    # Install NVIDIA TensorFlow plugins
    print("Installing NVIDIA PyIndex...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'nvidia-pyindex'])
    
    print("\nTensorFlow installation completed. Please verify GPU detection with check_gpu.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())