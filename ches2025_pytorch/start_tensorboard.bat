@echo off
title TensorBoard for CHES 2025

echo Starting TensorBoard for CHES 2025 training results...
echo.

rem Define the path to your Python virtual environment
rem This assumes your 'ches_env' folder is in the same directory as this .bat script.
set VENV_PATH=.\ches_env

rem --- Verify Virtual Environment Path ---
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo Error: Python virtual environment not found at "%VENV_PATH%".
    echo Please ensure:
    echo 1. Your virtual environment is named 'ches_env'.
    echo 2. The 'ches_env' folder is located in the same directory as this script.
    echo If your environment has a different name or location,
    echo please update the "set VENV_PATH=" line in this .bat file.
    pause
    exit /b 1
)

echo Available TensorBoard log directories in ./runs/:
rem Use '2>nul' to suppress "File Not Found" errors if the 'runs' directory doesn't exist yet
dir /b runs\ 2>nul || echo (No log directories found yet. Please run main_pytorch.py first to generate logs.)
echo.

echo Starting TensorBoard...
echo Open your browser to: http://localhost:6006/
echo (If this port is in use, TensorBoard will try another one and print it.)
echo.
echo Press Ctrl+C to stop TensorBoard.
echo.

rem --- Execute TensorBoard from the virtual environment ---
"%VENV_PATH%\Scripts\tensorboard.exe" --logdir=./runs/ --port 6006

echo.
echo TensorBoard session ended.
pause