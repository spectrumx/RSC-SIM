@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
  echo Usage: run_rfi_atms_batch.bat SENSOR_DIR
  echo   e.g. run_rfi_atms_batch.bat util\ATMS
  exit /b 1
)
set "SENSOR_DIR=%~1"
set "SCRIPT=ATMS_RFI_modeling.py"
echo Running ATMS RFI (5G + Starlink gateway) for all .nc4 in %SENSOR_DIR%
for %%F in ("%SENSOR_DIR%\*.nc4") do (
  set "nc4=%%F"
  echo --- %%~nxF
  python "%SCRIPT%" --sensor ATMS --nc4 "!nc4!" --out_dir "%SENSOR_DIR%"
)
echo Done.
