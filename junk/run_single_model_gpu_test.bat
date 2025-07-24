@echo off
echo Testing GPU functionality for TensorFlow with a single model...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Set environment variables for GPU
set TF_FORCE_GPU_ALLOW_GROWTH=true

:: Run GPU test script
python run_single_model_gpu_test.py

:: Keep console open
pause