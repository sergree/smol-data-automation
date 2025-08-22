#!/bin/bash

# Создаем директорию для логов если её нет
mkdir -p logs

# Создаем лог файл заранее
touch /var/log/cron.log

# Настраиваем cron расписание с полным путем к python
echo "${CRON_SCHEDULE:-0 */6 * * *} cd /app && /usr/local/bin/python -m src >> /var/log/cron.log 2>&1" | crontab -

# Запускаем cron в фоне
cron

# Показываем текущее расписание
echo "Установлено расписание cron:"
crontab -l

# Показываем логи в реальном времени
tail -f /var/log/cron.log