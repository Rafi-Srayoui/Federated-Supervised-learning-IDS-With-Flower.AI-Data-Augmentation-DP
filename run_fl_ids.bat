@echo off
REM ------------- user settings ---------------------------------
set NUM_CLIENTS=4
set PY=python      
REM -------------------------------------------------------------

setlocal EnableDelayedExpansion
set /a LAST=NUM_CLIENTS-1

echo Launching Flower server...
start "" cmd /k %PY% server.py
timeout /t 5 > nul

echo Launching %NUM_CLIENTS% clients...
for /l %%i in (0,1,!LAST!) do (
    start "" cmd /k %PY% client.py %%i
    timeout /t 1 > nul
)

echo All processes started.
endlocal
