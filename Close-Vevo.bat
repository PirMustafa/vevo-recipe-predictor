@echo off
REM ===================================================================
REM  Stops the Vevo server so it stops costing money.
REM
REM  Everything is preserved - models, environment, the application.
REM  Open-Vevo.bat will start it again in about two minutes.
REM ===================================================================
setlocal
set INSTANCE=i-01a1c4d84d57be632
set REGION=eu-north-1

title Vevo - shutting down

echo.
echo   Stopping the Vevo server...
echo.

aws ec2 stop-instances --instance-ids %INSTANCE% --region %REGION% --query "StoppingInstances[0].CurrentState.Name" --output text 2>nul
if errorlevel 1 (
  echo   ERROR: could not reach AWS. Check your credentials with: aws configure
  echo.
  pause
  exit /b 1
)

echo   Waiting for it to stop...
aws ec2 wait instance-stopped --instance-ids %INSTANCE% --region %REGION% >nul 2>&1

echo.
echo   -------------------------------------------------------------
echo    Server stopped. It is no longer being charged for.
echo.
echo    Nothing was lost. Open-Vevo.bat will bring it back in about
echo    two minutes.
echo   -------------------------------------------------------------
echo.
pause
