@echo off
chcp 65001 >nul

echo ========================================
echo   FinDynamicGraph_Agent - Run Experiment
echo ========================================
echo.

:: Activate venv
call .venv\Scripts\activate.bat

:: Run experiment
echo Running experiment...
python -m src.main_experiment
if errorlevel 1 (
    echo.
    echo ERROR: Experiment failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Experiment complete!
echo   Results saved to: data/experiments/
echo ========================================
echo.
pause
