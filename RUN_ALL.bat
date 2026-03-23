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
echo ======== overflowml plan 70 --compare ========
%OML% plan 70 --compare
echo.
echo ======== overflowml plan 140 --compare ========
%OML% plan 140 --compare
echo.
echo ======== overflowml can-run 40 ========
%OML% can-run 40
echo.
echo ======== overflowml can-run 140 ========
%OML% can-run 140
echo.
echo ======== overflowml benchmark ========
%OML% benchmark
echo.
echo ========================================
echo ALL COMMANDS COMPLETE - v0.9.0
echo ========================================
pause
