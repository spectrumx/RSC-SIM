@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
  echo Usage: run_rfi_ssmis_batch.bat SENSOR_DIR
  echo   e.g. run_rfi_ssmis_batch.bat util\SSMI-S
  exit /b 1
)
set "SENSOR_DIR=%~1"
set "SCRIPT=SSMI-S_RFI_modeling.py"
echo Running SSMI-S RFI (5G + Starlink gateway) for all input .nc4 in %SENSOR_DIR% (skips *_RFI.nc4; only SAID 285 / DMSP-F17)
for %%F in ("%SENSOR_DIR%\*.nc4") do (
  set "nx=%%~nxF"
  set "tail=!nx:~-8!"
  if /i "!tail!"=="_RFI.nc4" (
    echo --- Skipping RFI output: !nx!
  ) else (
    set "nc4=%%F"
    echo --- %%~nxF
    python "%SCRIPT%" --sensor SSMI-S --nc4 "!nc4!" --out_dir "%SENSOR_DIR%"
  )
)
echo Done.
