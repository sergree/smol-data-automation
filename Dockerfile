# Базовый образ Python 3.13 slim версии
FROM python:3.13-slim

# Устанавливаем cron для планирования задач и очищаем кэш apt
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию приложения
WORKDIR /app

# Копируем файл зависимостей и устанавливаем Python пакеты
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем весь исходный код приложения
COPY . .

# Копируем скрипт запуска и делаем его исполняемым
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Запускаем скрипт entrypoint.sh при старте контейнера
CMD ["/entrypoint.sh"]