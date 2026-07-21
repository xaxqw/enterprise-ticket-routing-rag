@echo off
chcp 65001 >nul 2>&1
title RAG Platform Launcher
cd /d D:\RAG
rag_proj_env_good\Scripts\python.exe start.py
pause
