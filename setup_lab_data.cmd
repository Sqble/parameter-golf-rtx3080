@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run setup_windows_env.cmd first.
  exit /b 1
)

".venv\Scripts\python.exe" data\cached_challenge_fineweb.py --variant sp1024 --train-shards 1
