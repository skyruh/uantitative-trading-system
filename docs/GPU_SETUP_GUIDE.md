# GPU Setup Guide for TensorFlow

This guide will help you set up your NVIDIA Quadro P2000 GPU to work with TensorFlow.

## Step 1: Install NVIDIA Drivers

1. Download the latest NVIDIA drivers for your Quadro P2000 from the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page.
   - Product Type: Quadro
   - Product Series: Quadro Series
   - Product: Quadro P2000
   - Operating System: Windows 11 64-bit

2. Install the downloaded driver and restart your computer.

3. Verify the driver installation by running `nvidia-smi` in Command Prompt:
   ```
   nvidia-smi
   ```
   This should display information about your GPU, including the driver version.

## Step 2: Install CUDA Toolkit

TensorFlow 2.15.0 requires CUDA 11.8.

1. Download CUDA Toolkit 11.8 from the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive) page.
   - Select Windows
   - Select x86_64
   - Select your Windows version
   - Select the exe (local) installer

2. Run the installer and follow these steps:
   - Choose "Custom" installation
   - Uncheck "Visual Studio Integration" if you don't need it
   - Complete the installation

3. Add CUDA to your PATH environment variable:
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click on "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Add the following paths:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
     ```

4. Create the following environment variables:
   - Variable: CUDA_PATH
   - Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

## Step 3: Install cuDNN

1. Download cuDNN 8.6 from the [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn) page.
   - You'll need to create a free NVIDIA account if you don't have one

2. Extract the downloaded zip file.

3. Copy the following files to your CUDA installation:
   - Copy `cuda\bin\*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - Copy `cuda\include\*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   - Copy `cuda\lib\x64\*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`

## Step 4: Install TensorFlow with GPU Support

1. Open Command Prompt and activate your virtual environment:
   ```
   call venv\Scripts\activate.bat
   ```

2. Install TensorFlow 2.10.0 (which is compatible with CUDA 11.8):
   ```
   pip uninstall -y tensorflow tensorflow-gpu
   pip install tensorflow==2.10.0
   ```

## Step 5: Verify GPU Setup

1. Run the following Python code to verify that TensorFlow can detect your GPU:
   ```python
   import tensorflow as tf
   print("TensorFlow version:", tf.__version__)
   print("GPU available:", tf.config.list_physical_devices('GPU'))
   ```

2. If your GPU is properly set up, you should see output like:
   ```
   TensorFlow version: 2.10.0
   GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```

## Step 6: Run the GPU Test

1. Run the GPU test script:
   ```
   python check_gpu.py
   ```

2. If successful, you should see output indicating that TensorFlow is using your GPU.

## Troubleshooting

If TensorFlow still doesn't detect your GPU:

1. Make sure your NVIDIA drivers are up to date.
2. Verify that CUDA and cuDNN are properly installed.
3. Check that the environment variables are correctly set.
4. Try restarting your computer.
5. Make sure your GPU is not being used by another application.

## Additional Resources

- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)