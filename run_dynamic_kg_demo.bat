@echo off
setlocal enabledelayedexpansion

REM Run this file from the repository root.
set TICKER=0700.HK
set DATE_A=2025-01-01
set DATE_B=2025-03-01
set OUT_DIR=data\experiments
set SNAPSHOT_A=%OUT_DIR%\snapshot_demo_A.json
set SNAPSHOT_B=%OUT_DIR%\snapshot_demo_B.json

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

REM ── Load .env into shell environment ────────────────────────────────
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            set "%%A=%%B"
        )
    )
)

REM ── Pre-flight checks ──────────────────────────────────────────────
echo.
echo ============================================================
echo  Pre-flight Checks
echo ============================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ and add it to PATH.
    exit /b 1
)
echo  [OK] Python found

REM Check Neo4j env vars
if "%NEO4J_URI%"=="" (
    echo [WARN] NEO4J_URI not set. Defaulting to neo4j://localhost:7687
    set NEO4J_URI=neo4j://localhost:7687
)
if "%NEO4J_USER%"=="" (
    set NEO4J_USER=neo4j
)
if "%NEO4J_PASSWORD%"=="" (
    echo [WARN] NEO4J_PASSWORD not set. Defaulting to "neo4j"
    set NEO4J_PASSWORD=neo4j
)

REM Quick Neo4j connectivity probe
python -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('%NEO4J_URI%', auth=('%NEO4J_USER%', '%NEO4J_PASSWORD%')); d.verify_connectivity(); d.close(); print('OK')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot connect to Neo4j at %NEO4J_URI%.
    echo         Please ensure Neo4j is running and NEO4J_URI / NEO4J_PASSWORD are set.
    exit /b 1
)
echo  [OK] Neo4j connected

if not exist %OUT_DIR% mkdir %OUT_DIR%

echo.
echo ============================================================
echo  1. Inspect point-in-time graph: %TICKER% at %DATE_A%
echo ============================================================
python -m src.scripts.inspect_snapshot --ticker %TICKER% --date %DATE_A%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  2. Inspect point-in-time graph: %TICKER% at %DATE_B%
echo ============================================================
python -m src.scripts.inspect_snapshot --ticker %TICKER% --date %DATE_B%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  3. Export both snapshots
echo ============================================================
python -m src.scripts.export_graph_snapshot --ticker %TICKER% --as-of %DATE_A% --output %SNAPSHOT_A%
if errorlevel 1 goto :failed
python -m src.scripts.export_graph_snapshot --ticker %TICKER% --as-of %DATE_B% --output %SNAPSHOT_B%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  4. Diff snapshots
echo ============================================================
python -m src.scripts.diff_snapshots --a %SNAPSHOT_A% --b %SNAPSHOT_B%
REM diff_snapshots exits 1 when snapshots are DIFFERENT (expected),
REM and 0 when identical.  We treat exit code 1 as success here.
if errorlevel 2 goto :failed
if errorlevel 1 (
    echo.
    echo  [PASS] Temporal divergence detected. The two graph states are different.
    exit /b 0
)

echo.
echo  [NOTE] The snapshots were identical. Choose two dates with more market changes.
exit /b 0

:failed
echo.
echo  [ERROR] Demo stopped. Check the output above for details.
exit /b 1
