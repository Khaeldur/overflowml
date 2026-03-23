@echo off
set OML=C:\Users\mvait\Documents\ComfyUI\.venv\Scripts\overflowml.exe

echo.
echo ======== overflowml detect ========
%OML% detect
echo.
echo ======== overflowml doctor ========
%OML% doctor
echo.
echo ======== overflowml plan 40 --compare ========
%OML% plan 40 --compare
echo.
echo ======== overflowml can-run 40 ========
%OML% can-run 40
echo.
echo ======== overflowml can-run 10000 ========
%OML% can-run 10000
echo.
echo ======== overflowml benchmark ========
%OML% benchmark
echo.
echo ========================================
echo ALL COMMANDS COMPLETE
echo ========================================
pause
