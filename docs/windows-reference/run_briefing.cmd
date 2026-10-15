@echo off
chcp 65001 >nul
rem 用法：run_briefing.cmd [活动日期] ["发布时间"]，例如 run_briefing.cmd 2026-10-04 "2026-10-04 08:00"
rem 不带参数时按最近的 08/16/20 时次和对应活动日自动生成
set "ARGS="
if not "%~1"=="" set "ARGS=--date %~1"
if not "%~2"=="" set "ARGS=%ARGS% --issue "%~2""
wsl.exe -d Ubuntu --cd /mnt/c/Users/chengyang/Documents/Codex/2026-10-03/xi-w/work -- /home/chengyang/.venvs/ec-gfs-server/bin/python event_briefing.py %ARGS%
pause
