@echo off
echo Setting up TensorFlow with GPU support...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Set environment variables for GPU
echo Setting environment variables for GPU...
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async

:: Check NVIDIA driver
echo Checking NVIDIA driver...
nvidia-smi

:: Install TensorFlow with GPU support
echo Installing TensorFlow with GPU support...
python install_tf_gpu.py

:: Run GPU test
echo.
echo Running GPU test...
python check_gpu.py

echo.
echo If the GPU test was successful, you can now run the full training with:
echo .\run_lstm_direct.bat

:: Keep console open
pause