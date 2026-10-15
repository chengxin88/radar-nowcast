@echo off
chcp 65001 >nul
title 灯光秀研判 一键发报
echo 正在抓取最新资料、出稿并推送到微信（提醒、详细版、简版三条），约半分钟到两分钟……
echo.
wsl.exe -d Ubuntu -- bash /mnt/c/Users/chengyang/Documents/Codex/2026-10-03/xi-w/work/push_now.sh
if errorlevel 1 (
  echo.
  echo [失败] 没有推送成功。请先在微信里给机器人发一句话，再双击重试；仍失败请看 work\logs\push_YYYYMM.log。
) else (
  echo.
  echo [完成] 已推送到微信。稿件保存在 outputs 文件夹。
)
echo.
pause