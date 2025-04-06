@echo off
:start
python app.py
timeout /t 5
ngrok http 5000
goto start 