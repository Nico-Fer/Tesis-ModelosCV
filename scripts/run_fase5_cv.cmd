@echo off
rem Lanza o REANUDA la validacion cruzada completa de la Fase 5.
rem Doble clic en este archivo (o ejecutarlo desde cualquier terminal). Tras un corte de
rem luz se vuelve a ejecutar el MISMO archivo: retoma desde el ultimo checkpoint.
rem Ver docs\fase5_cv_bitacora.md.
cd /d "%~dp0.."
title Fase 5 - validacion cruzada (reanudable)
venv-torch\Scripts\python.exe scripts\run_fase5_cv.py
echo.
echo La corrida termino o se interrumpio (codigo %ERRORLEVEL%).
echo Para reanudar, volver a ejecutar scripts\run_fase5_cv.cmd
pause
