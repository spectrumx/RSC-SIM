@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
  echo Usage: run_rfi_amsua_batch.bat SENSOR_DIR
  echo   e.g. run_rfi_amsua_batch.bat util\AMSU-A
  exit /b 1
)
set "SENSOR_DIR=%~1"
set "SCRIPT=AMSU-A_RFI_modeling.py"
echo Running AMSU-A RFI (5G + Starlink gateway) for all input .nc4 in %SENSOR_DIR% (skips *_RFI.nc4)
for %%F in ("%SENSOR_DIR%\*.nc4") do (
  set "nx=%%~nxF"
  set "tail=!nx:~-8!"
  if /i "!tail!"=="_RFI.nc4" (
    echo --- Skipping RFI output: !nx!
  ) else (
    set "nc4=%%F"
    echo --- %%~nxF
    python "%SCRIPT%" --sensor AMSU-A --nc4 "!nc4!" --out_dir "%SENSOR_DIR%"
  )
)
echo Done.
