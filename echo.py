import logging
import requests
from aiogram import Bot, Dispatcher, executor, types
from settings import API_TOKEN
from telegram.ext import Updater, CommandHandler, MessageHandler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Инициализация модели и токенизатора ESGify
model_name = "ai-lab/ESGify"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def esgify(text):
    """Функция для анализа текста моделью ESGify."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.nn.functional.softmax(logits, dim=1)
    return scores

def start(update, context):
    """Отправляет приветственное сообщение."""
    update.message.reply_text('Привет! Отправьте мне текст для анализа ESG.')

def handle_message(update, context):
    """Обрабатывает полученное сообщение и возвращает результат ESG анализа."""
    text = update.message.text
    result = esgify(text)
    update.message.reply_text(f'Результат ESG анализа: {result}')

def main():
    """Основная функция для запуска бота."""
    updater = Updater(API_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

