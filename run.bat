@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto setup_error
)
if not exist ".venv\Scripts\streamlit.exe" (
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto setup_error
)
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
goto :eof

:setup_error
echo Python 설치 또는 패키지 설치에 실패했습니다. 설치 가이드를 확인해 주세요.
pause
