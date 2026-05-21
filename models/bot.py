import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ConversationHandler, ContextTypes
)
from catboost import CatBoostRegressor

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


BASE_DIR = Path(__file__).parent.parent
model = CatBoostRegressor()
model.load_model(BASE_DIR / "models" / "catboost_model")  # имя файла подправь под своё


AREA, ROOMS, FLOOR, FLOORS_TOTAL, DISTRICT = range(5)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я помогу оценить стоимость квартиры.\n\n"
        "Введи общую площадь квартиры (м²):\n"
        "Например: 54.5"
    )
    return AREA

async def get_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        area = float(update.message.text.replace(",", "."))
        context.user_data["area"] = area
        await update.message.reply_text("Сколько комнат? (введи число, например: 2)")
        return ROOMS
    except ValueError:
        await update.message.reply_text("❌ Введи число, например: 54.5")
        return AREA

async def get_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        rooms = int(update.message.text)
        context.user_data["rooms"] = rooms
        await update.message.reply_text("На каком этаже квартира?")
        return FLOOR
    except ValueError:
        await update.message.reply_text("❌ Введи целое число, например: 2")
        return ROOMS

async def get_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        floor = int(update.message.text)
        context.user_data["floor"] = floor
        await update.message.reply_text("Сколько этажей в доме?")
        return FLOORS_TOTAL
    except ValueError:
        await update.message.reply_text("❌ Введи целое число")
        return FLOOR

async def get_floors_total(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        floors_total = int(update.message.text)
        context.user_data["floors_total"] = floors_total

        keyboard = [["Центр", "Север"], ["Юг", "Восток"], ["Запад", "Пригород"]]
        await update.message.reply_text(
            "Выбери район:",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        )
        return DISTRICT
    except ValueError:
        await update.message.reply_text("❌ Введи целое число")
        return FLOORS_TOTAL

async def get_district(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["district"] = update.message.text

    data = context.user_data
    features = [[
        data["area"],
        data["rooms"],
        data["floor"],
        data["floors_total"],
        data["district"],
    ]]

    try:
        price = model.predict(features)[0]
        price_low = price * 0.92
        price_high = price * 1.08

        await update.message.reply_text(
            f"🏠 <b>Оценка квартиры</b>\n\n"
            f"📐 Площадь: {data['area']} м²\n"
            f"🚪 Комнат: {data['rooms']}\n"
            f"🏢 Этаж: {data['floor']} из {data['floors_total']}\n"
            f"📍 Район: {data['district']}\n\n"
            f"💰 <b>Оценочная стоимость:</b>\n"
            f"от <b>{price_low:,.0f}</b> до <b>{price_high:,.0f}</b> ₽\n"
            f"(~{price:,.0f} ₽)",
            parse_mode="HTML",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка предсказания: {e}")

    await update.message.reply_text("Хочешь оценить ещё одну квартиру? /start")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отменено. /start — начать заново.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AREA:         [MessageHandler(filters.TEXT & ~filters.COMMAND, get_area)],
            ROOMS:        [MessageHandler(filters.TEXT & ~filters.COMMAND, get_rooms)],
            FLOOR:        [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floor)],
            FLOORS_TOTAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floors_total)],
            DISTRICT:     [MessageHandler(filters.TEXT & ~filters.COMMAND, get_district)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
