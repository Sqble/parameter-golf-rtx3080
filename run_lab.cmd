@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run setup_windows_env.cmd first.
  exit /b 1
)

".venv\Scripts\python.exe" lab\run_experiment.py %*
