#!/bin/bash


# Налаштування
APP_NAME="main.py"           # або "uvicorn main:app" якщо запускаєте через uvicorn
LOG_FILE="logs/api.log"

# Створити папку logs, якщо не існує
mkdir -p logs


# Пошук PID для main.py або uvicorn main:app
echo "🔎 Пошук процесу $APP_NAME..."
PID=$(ps aux | grep "$APP_NAME" | grep -v grep | grep -v restart.sh | awk '{print $2}')


if [ -n "$PID" ]; then
  echo "🛑 Зупинка процесу PID=$PID"
  kill $PID
  sleep 2
  # Перевірка, чи процес ще живий
  if ps -p $PID > /dev/null; then
    echo "⚠️  kill не спрацював, застосовую kill -9"
    kill -9 $PID
    sleep 1
  fi
else
  echo "ℹ️ Процес не знайдено, запускаємо новий"
fi


# echo "🗄️ Ініціалізація бази даних..." # (розкоментуйте якщо потрібно)
# python3 -m app.db.init_db 2>&1 | tee -a $LOG_FILE


echo "🚀 Запуск $APP_NAME..."
if [[ "$APP_NAME" == *uvicorn* ]]; then
  nohup $APP_NAME > $LOG_FILE 2>&1 &
else
  nohup python3 $APP_NAME > $LOG_FILE 2>&1 &
fi


NEW_PID=$!
echo "✅ Новий процес запущено з PID=$NEW_PID"
echo "📜 Логи: tail -f $LOG_FILE"