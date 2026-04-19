@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
  echo Usage: run_rfi_ssmis_batch.bat SENSOR_DIR
  echo   e.g. run_rfi_ssmis_batch.bat util\SSMI-S
  exit /b 1
)
set "SENSOR_DIR=%~1"
set "SCRIPT=SSMI-S_RFI_modeling.py"
echo Running SSMI-S RFI (5G + Starlink gateway) for all .nc4 in %SENSOR_DIR% (only SAID 285 / DMSP-F17)
for %%F in ("%SENSOR_DIR%\*.nc4") do (
  set "nc4=%%F"
  echo --- %%~nxF
  python "%SCRIPT%" --sensor SSMI-S --nc4 "!nc4!" --out_dir "%SENSOR_DIR%"
)
echo Done.
