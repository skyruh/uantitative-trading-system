#!/usr/bin/env python3
"""
Script to check GPU availability and performance for TensorFlow.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Set TensorFlow logging level to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_gpu_availability():
    """Check if GPU is available for TensorFlow."""
    print("=" * 50)
    print("TensorFlow GPU Check")
    print("=" * 50)
    print("TensorFlow version:", tf.__version__)
    
    # Check physical devices
    print("\nPhysical devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device.device_type}: {device.name}")
    
    # Check GPU devices specifically
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\nNo GPU found. TensorFlow will use CPU.")
        print("\nPossible reasons:")
        print("1. CUDA and cuDNN are not properly installed")
        print("2. TensorFlow version is not compatible with your CUDA version")
        print("3. GPU drivers need to be updated")
        print("\nCurrent CUDA paths:")
        for env_var in ['CUDA_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']:
            print(f"  {env_var}: {os.environ.get(env_var, 'Not set')}")
        return False
    
    print(f"\nFound {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        try:
            # Get device details
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
            
            # Configure memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Memory growth set to True for GPU {i}")
        except RuntimeError as e:
            print(f"  Error configuring GPU {i}: {e}")
    
    return len(gpus) > 0

def run_gpu_test():
    """Run a simple test to verify GPU performance."""
    print("\nRunning GPU performance test...")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Generate random data
    x = np.random.random((1000, 1000))
    y = np.random.random((1000, 10))
    
    # Time the training
    start_time = time.time()
    model.fit(x, y, epochs=5, batch_size=64, verbose=2)
    end_time = time.time()
    
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Check device placement
    print("\nDevice placement for operations:")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        
        print(f"Matrix multiplication result shape: {c.shape}")
        print(f"Tensor device: {c.device}")
        if 'GPU' in c.device:
            print("✓ Operation executed on GPU")
        else:
            print("✗ Operation executed on CPU")
    except Exception as e:
        print(f"Error during device placement test: {e}")
    
    return True

def main():
    """Main function to check GPU and run tests."""
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        # Run performance test
        run_gpu_test()
        print("\nGPU is properly configured for TensorFlow!")
    else:
        print("\nNo GPU available. Training will be slower on CPU.")
        # Still run the test on CPU for comparison
        run_gpu_test()
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()