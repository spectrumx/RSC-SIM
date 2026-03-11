@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
  echo Usage: run_rfi_amsua_batch.bat SENSOR_DIR
  echo   e.g. run_rfi_amsua_batch.bat util\AMSU-A
  exit /b 1
)
set "SENSOR_DIR=%~1"
set "SCRIPT=AMSU-A_RFI_modeling.py"
echo Running AMSU-A RFI for all .nc4 in %SENSOR_DIR%
for %%F in ("%SENSOR_DIR%\*.nc4") do (
  set "nc4=%%F"
  echo --- %%~nxF
  python "%SCRIPT%" --sensor AMSU-A --nc4 "!nc4!" --out_dir "%SENSOR_DIR%"
)
echo Done.
