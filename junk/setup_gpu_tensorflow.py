#!/usr/bin/env python3
"""
Script to set up TensorFlow with GPU support.
"""

import os
import sys
import subprocess
import platform
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPUSetup")

def check_gpu_with_nvidia_smi():
    """Check GPU using nvidia-smi command."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected with nvidia-smi:")
            logger.info(result.stdout)
            return True
        else:
            logger.error("nvidia-smi command failed:")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error running nvidia-smi: {str(e)}")
        return False

def check_cuda_installation():
    """Check CUDA installation."""
    try:
        # Check CUDA_PATH environment variable
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            logger.info(f"CUDA_PATH environment variable found: {cuda_path}")
            if os.path.exists(cuda_path):
                logger.info(f"CUDA installation directory exists: {cuda_path}")
                return True
            else:
                logger.warning(f"CUDA_PATH directory does not exist: {cuda_path}")
        else:
            logger.warning("CUDA_PATH environment variable not found")
        
        # Check common CUDA installation paths
        common_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"Found CUDA installation at: {path}")
                return True
        
        logger.warning("No CUDA installation found in common paths")
        return False
    except Exception as e:
        logger.error(f"Error checking CUDA installation: {str(e)}")
        return False

def check_cudnn_installation():
    """Check cuDNN installation."""
    try:
        cuda_path = os.environ.get('CUDA_PATH')
        if not cuda_path:
            logger.warning("CUDA_PATH environment variable not found, cannot check cuDNN")
            return False
        
        cudnn_path = os.path.join(cuda_path, 'include', 'cudnn.h')
        if os.path.exists(cudnn_path):
            logger.info(f"cuDNN found at: {cudnn_path}")
            return True
        else:
            logger.warning(f"cuDNN not found at: {cudnn_path}")
            return False
    except Exception as e:
        logger.error(f"Error checking cuDNN installation: {str(e)}")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow can detect GPU."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name}")
            return True
        else:
            logger.warning("TensorFlow did not detect any GPUs")
            return False
    except Exception as e:
        logger.error(f"Error checking TensorFlow GPU: {str(e)}")
        return False

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support."""
    try:
        logger.info("Installing TensorFlow with GPU support...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.10.0'], check=True)
        logger.info("TensorFlow with GPU support installed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing TensorFlow with GPU support: {str(e)}")
        return False

def main():
    """Main function to set up TensorFlow with GPU support."""
    logger.info("Starting GPU setup for TensorFlow...")
    
    # Check system information
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Operating system: {platform.platform()}")
    
    # Check NVIDIA GPU
    gpu_detected = check_gpu_with_nvidia_smi()
    if not gpu_detected:
        logger.error("No NVIDIA GPU detected. Please install NVIDIA drivers.")
        return 1
    
    # Check CUDA installation
    cuda_installed = check_cuda_installation()
    if not cuda_installed:
        logger.error("CUDA not properly installed. Please install CUDA 11.2 or compatible version.")
        return 1
    
    # Check cuDNN installation
    cudnn_installed = check_cudnn_installation()
    if not cudnn_installed:
        logger.warning("cuDNN not properly installed. This may affect TensorFlow performance.")
    
    # Check TensorFlow GPU
    tf_gpu = check_tensorflow_gpu()
    if not tf_gpu:
        logger.warning("TensorFlow is not detecting GPU. Attempting to reinstall TensorFlow with GPU support...")
        install_tensorflow_gpu()
        
        # Check again
        tf_gpu = check_tensorflow_gpu()
        if not tf_gpu:
            logger.error("TensorFlow still not detecting GPU after reinstallation.")
            logger.error("Please ensure NVIDIA drivers, CUDA, and cuDNN are properly installed.")
            return 1
    
    logger.info("GPU setup for TensorFlow completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())