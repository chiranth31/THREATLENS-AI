@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo       THREATLENS AI - SECURITY OPERATIONS
 echo ================================================
if not exist models\model_bundle.json (
  echo.
  echo [WARNING] models\model_bundle.json is missing.
  echo Run the Google Colab notebook and copy the exported model files into models\
  echo The login/dashboard will still start, but URL ML scanning will remain disabled.
  echo.
)
if not exist .venv\Scripts\python.exe (
  echo Creating local Python environment...
  py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --disable-pip-version-check -q -r requirements.txt
python backend\app.py
