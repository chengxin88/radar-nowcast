@echo off
chcp 65001 >nul
title 市民中心研判 一键发报
echo 正在抓取雷达、闪电、自动站资料，出稿并推送到微信，约半分钟……
wsl.exe -d Ubuntu -- bash /mnt/c/Users/chengyang/Documents/Codex/2026-10-03/xi-w/work/push_now_civic.sh
echo.
pause
