@echo off
echo.
echo === OverflowML PyPI Upload ===
echo.
set /p "TOKEN=Paste your PyPI API token (starts with pypi-): "
echo.
echo Uploading to PyPI...
"C:\Users\mvait\Documents\ComfyUI\.venv\Scripts\python.exe" -m twine upload dist/* -u __token__ -p %TOKEN%
echo.
if errorlevel 1 (
    echo [ERROR] Upload failed. Check your token.
) else (
    echo [OK] Published! Install with: pip install overflowml
)
pause
