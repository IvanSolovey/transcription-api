#!/bin/bash

APP_NAME="main.py"
LOG_FILE="logs/api.log"

echo "🔎 Пошук процесу $APP_NAME..."
PID=$(ps aux | grep "$APP_NAME" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
  echo "🛑 Зупинка процесу PID=$PID"
  kill -9 $PID
  sleep 1
else
  echo "ℹ️ Процес не знайдено, запускаємо новий"
fi

echo "�️ Ініціалізація бази даних..."
python3 -m app.db.init_db 2>&1 | tee -a $LOG_FILE

echo "�🚀 Запуск $APP_NAME..."
nohup python3 $APP_NAME > $LOG_FILE 2>&1 &

NEW_PID=$!
echo "✅ Новий процес запущено з PID=$NEW_PID"
echo "📜 Логи: tail -f $LOG_FILE"