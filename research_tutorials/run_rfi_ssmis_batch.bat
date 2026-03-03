@echo off
setlocal enabledelayedexpansion
if "%~1"=="" (
  echo Usage: run_rfi_ssmis_batch.bat SATELLITE_DIR
  echo   e.g. run_rfi_ssmis_batch.bat util\DMSP-F17
  exit /b 1
)
set "SAT_DIR=%~1"
for %%A in ("%SAT_DIR%") do set "SAT_NAME=%%~nxA"
set "SCRIPT=SSMI-S_RFI_modeling.py"
echo Running SSMI-S RFI for all .nc4 in %SAT_DIR% (satellite: %SAT_NAME%)
for %%F in ("%SAT_DIR%\*.nc4") do (
  set "nc4=%%F"
  set "stem=%%~nF"
  set "ecef=%SAT_DIR%\%SAT_NAME%_ECEF_lookup_!stem!.csv"
  if exist "!ecef!" (
    echo --- %%~nxF
    python "%SCRIPT%" --sat %SAT_NAME% --nc4 "!nc4!" --ecef "!ecef!" --out_dir "%SAT_DIR%"
  ) else (
    echo Skip %%F: ECEF not found: !ecef!
  )
)
echo Done.
