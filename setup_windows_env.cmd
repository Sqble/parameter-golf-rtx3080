@echo off
setlocal

py -3.12 -m venv .venv
if errorlevel 1 exit /b %errorlevel%

call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b %errorlevel%

python -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

python -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%

python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
exit /b %errorlevel%
