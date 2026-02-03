@echo off
REM Start All Components Script for Windows
REM This script starts the FL server, multiple clients, and the web interface

echo ================================
echo Starting Federated Learning System
echo ================================

REM Start FL Server in new window
echo Starting FL Server...
start "FL Server" cmd /k python run_server.py --num-rounds 10 --min-clients 2
timeout /t 5 /nobreak

REM Start Clients
set NUM_CLIENTS=5
echo Starting %NUM_CLIENTS% clients...
for /L %%i in (0,1,4) do (
    echo   Starting Client %%i...
    start "FL Client %%i" cmd /k python run_client.py --client-id %%i --num-clients %NUM_CLIENTS%
    timeout /t 2 /nobreak
)

REM Start Web Interface
echo Starting Web Interface...
start "Web Interface" cmd /k python run_web.py

echo.
echo ================================
echo All components started!
echo ================================
echo FL Server:       Running
echo Clients:         %NUM_CLIENTS% running
echo Web Interface:   http://localhost:5000
echo.
echo Close the command windows to stop components
echo ================================

pause
