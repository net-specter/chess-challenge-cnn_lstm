@echo off
echo Starting TensorBoard for CNN-LSTM training results...
echo.
echo Available log directories:
dir /b Result\*tensorboard* 2>nul
echo.
echo Starting TensorBoard...
echo Open your browser to: http://localhost:6006/
echo Press Ctrl+C to stop TensorBoard
echo.
tensorboard --logdir=./Result/
pause
