@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   FinDynamicGraph_Agent - Windows Setup
echo ========================================
echo.

:: 1. Check Python
echo [1/5] Checking Python 3.11+...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python %PYVER% found.

:: Check version >= 3.11
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% LSS 3 (
    echo ERROR: Python 3.11+ required, found %PYVER%
    pause
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 11 (
    echo ERROR: Python 3.11+ required, found %PYVER%
    pause
    exit /b 1
)
echo Version check passed.
echo.

:: 2. Create venv
echo [2/5] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

:: 3. Activate venv and install dependencies
echo [3/5] Installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed.
echo.

:: 4. Check .env
echo [4/5] Checking .env...
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create .env with the following content:
    echo.
    echo   NEO4J_URI=neo4j://localhost:7687
    echo   NEO4J_USER=neo4j
    echo   NEO4J_PASSWORD=testpassword
    echo   TAVILY_API_KEY=your_key_here
    echo   LLM_API_BASE=your_base_here
    echo   LLM_API_KEY=your_key_here
    echo   LLM_MODEL=your_model_here
    echo.
    echo After creating .env, run this script again.
    pause
    exit /b 1
) else (
    echo .env found.
)
echo.

:: 5. Run collection
echo [5/5] Running data collection...
echo.
python -m src.main_collect
if errorlevel 1 (
    echo.
    echo WARNING: Collection had errors (some data may have been fetched).
)
echo.
echo ========================================
echo   Collection complete!
echo   Data saved to: data/raw/
echo   KG data written to Neo4j.
echo ========================================
echo.
echo Next step: python -m src.main_experiment
echo.
pause
