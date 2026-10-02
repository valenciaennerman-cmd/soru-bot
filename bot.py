import logging
import time
import json
import re
import os
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import threading
from flask import Flask

load_dotenv()

keep_alive_app = Flask(__name__)
@keep_alive_app.route('/')
def home():
    return "Bot is running!"

def run_flask():
    keep_alive_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
except ValueError:
    ALLOWED_USER_ID = 0
NVIDIA_API_KEY_1 = os.getenv("NVIDIA_API_KEY_1")
NVIDIA_API_KEY_2 = os.getenv("NVIDIA_API_KEY_2")
NVIDIA_MODEL_PRIMARY = os.getenv("NVIDIA_MODEL_PRIMARY", "deepseek-ai/deepseek-v4-pro")
NVIDIA_MODEL_FALLBACK = os.getenv("NVIDIA_MODEL_FALLBACK", "meta/muse-glimmer-30b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-20b")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Yazim kurallari sozlugunu (Cache) bellege yukle
YAZIM_SOZLUGU = {}
try:
    with open("yazim.json", "r", encoding="utf-8") as f:
        YAZIM_SOZLUGU = json.load(f)
    logger.info(f"{len(YAZIM_SOZLUGU)} adet yazim kurali bellege alindi.")
except Exception as e:
    logger.error(f"Yazim sozlugu yuklenemedi: {e}")

def duzelt_metin(metin: str) -> str:
    """Soruyu API'ye yollamadan once yazim hatalarini hizlica duzeltir (Regex tabanli, case-insensitive)"""
    if not YAZIM_SOZLUGU:
        return metin
    # Sadece tam kelime eslesmelerini degistirmek icin regex kullaniyoruz
    for yanlis, dogru in YAZIM_SOZLUGU.items():
        pattern = re.compile(rf"\b{yanlis}\b", re.IGNORECASE)
        metin = pattern.sub(dogru, metin)
    return metin

SYSTEM_PROMPT = (
    "Akilli saat ekraninda okunan kisa Q&A asistanisin. "
    "Kurallar: sadece duz metin, markdown/emoji yok. "
    "Kullanici uzun cevap istemedikce kisa tut. Giris cumlesi kurma, dogrudan cevapla. "
    "ÖNEMLİ KURAL: Matematik ve mantık sorularını doğru çözmek için mutlaka önce adım adım düşünmelisin. Tüm düşüncelerini, hesaplamalarını ve ara işlemlerini <dusunce> ve </dusunce> etiketleri arasına yaz. Bu etiketlerin dışına (en sona) SADECE bulduğun net cevabı çok kısa bir şekilde yaz. Akıllı saat ekranı dar olduğu için <dusunce> kısmı kullanıcıdan gizlenecektir. "
    "GÖRSEL SORU KURALI: Eğer kullanıcı sana bir soru atarsa ve soruda 'şekildeki', 'yandaki grafikte', 'görsele göre' gibi ifadelere atıf varsa AMA o görselin detayları metinde EKSİKSE, ASLA kafadan uydurma veya varsayım yapma. Bunun yerine kullanıcıya dönüp eksik olan görsel bilgiyi sor. Örn: 'Şekildeki üçgenin açıları kaç derece?' veya 'Tablodaki değerleri okur musun?'. Kullanıcı eksik bilgiyi verince soruyu çöz. "
    "TYT Turkce / Yazim Kurallari soruldugunda TDK'ye gore 'Ayri yazilir: ...' veya 'Birlesik yazilir: ...' seklinde TAK diye kisa ve net dogru cevabi ver. "
    "Emin degilsen uydurma, bilmedigini soyle. "
    "Soru hangi dildeyse o dilde cevap ver."
)

USER_HISTORIES = {}

def update_history(user_id: int, role: str, content: str) -> list:
    if user_id not in USER_HISTORIES:
        USER_HISTORIES[user_id] = []
    USER_HISTORIES[user_id].append({"role": role, "content": content})
    # Son 6 mesaji (3 soru-cevap) tut
    if len(USER_HISTORIES[user_id]) > 6:
        USER_HISTORIES[user_id] = USER_HISTORIES[user_id][-6:]
    return USER_HISTORIES[user_id]

def clean_response(content: str) -> str:
    # <dusunce> ... </dusunce> bloklarini metinden tamamen temizle
    cleaned = re.sub(r'<dusunce>.*?</dusunce>', '', content, flags=re.DOTALL).strip()
    return cleaned if cleaned else content.strip()

def ask_nvidia(chat_history: list, model: str, api_key: str) -> str:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=180.0
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("EmptyContent")
            return clean_response(content)
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "429" in error_str or "EmptyContent" in error_str or "timeout" in error_str.lower()) and attempt < max_retries - 1:
                logger.warning(f"NVIDIA API hatasi veya bos cevap, {attempt+1}. deneme basarisiz. 5 sn sonra tekrar deneniyor...")
                time.sleep(5)
                continue
            raise e

def ask_deepseek(chat_history: list, model: str, api_key: str) -> str:
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
        timeout=180.0
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("EmptyContent")
            return clean_response(content)
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "429" in error_str or "EmptyContent" in error_str or "timeout" in error_str.lower()) and attempt < max_retries - 1:
                logger.warning(f"DeepSeek API hatasi veya bos cevap, {attempt+1}. deneme basarisiz. 5 sn sonra tekrar deneniyor...")
                time.sleep(5)
                continue
            raise e

def ask_gemini(chat_history: list, model: str, api_key: str) -> str:
    gemini_client = genai.Client(api_key=api_key)
    
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=800,
                ),
            )
            content = response.text
            if not content or not content.strip():
                raise ValueError("EmptyContent")
            return clean_response(content)
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "429" in error_str or "EmptyContent" in error_str) and attempt < max_retries - 1:
                logger.warning(f"Gemini API hatasi veya bos cevap, {attempt+1}. deneme basarisiz. 5 sn sonra tekrar deneniyor...")
                time.sleep(5)
                continue
            raise e

def ask_gemini_vision(question: str, image_bytes: bytearray, api_key: str) -> str:
    gemini_client = genai.Client(api_key=api_key)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    question,
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                ],
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=800,
                ),
            )
            content = response.text
            if not content or not content.strip():
                raise ValueError("EmptyContent")
            return clean_response(content)
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "429" in error_str or "EmptyContent" in error_str) and attempt < max_retries - 1:
                logger.warning(f"Gemini API (Vision) hatasi veya bos cevap, {attempt+1}. deneme basarisiz. 5 sn sonra tekrar deneniyor...")
                time.sleep(5)
                continue
            raise e

def ask_with_fallback(chat_history: list) -> str:
    # 1. Deneme: DeepSeek Native (En kararli)
    # 2. Deneme: Gemini API + Gemini 3.8 Flash
    # 3. Deneme: API Key 1 + NVIDIA DeepSeek
    # 4. Deneme: API Key 2 + Muse Glimmer
    strategies = [
        {"provider": "deepseek", "key": DEEPSEEK_API_KEY, "model": DEEPSEEK_MODEL, "desc": "DeepSeek Native API"},
        {"provider": "gemini", "key": GEMINI_API_KEY, "model": GEMINI_MODEL, "desc": "Gemini API + Gemini 3.8 Flash"},
        {"provider": "nvidia", "key": NVIDIA_API_KEY_1, "model": NVIDIA_MODEL_PRIMARY, "desc": "API 1 + NVIDIA DeepSeek"},
        {"provider": "nvidia", "key": NVIDIA_API_KEY_2, "model": NVIDIA_MODEL_FALLBACK, "desc": "API 2 + Muse Glimmer"}
    ]
    
    last_error = None
    
    for attempt, strategy in enumerate(strategies):
        api_key = strategy["key"]
        model = strategy["model"]
        provider = strategy["provider"]
        
        if not api_key or "BURAYA" in api_key:
            logger.warning(f"{strategy['desc']} icin API anahtari tanimsiz, atliyorum.")
            continue

        try:
            logger.info(f"Deneyelen strateji: {strategy['desc']}")
            if provider == "deepseek":
                return ask_deepseek(chat_history, model, api_key)
            elif provider == "nvidia":
                return ask_nvidia(chat_history, model, api_key)
            elif provider == "gemini":
                return ask_gemini(chat_history, model, api_key)
        except Exception as e:
            last_error = e
            logger.warning(f"Strateji '{strategy['desc']}' basarisiz oldu: {e}")
            if attempt < len(strategies) - 1:
                logger.info("Diger API ve modele geciliyor...")
                time.sleep(2)
                
    logger.error(f"Tum denemeler basarisiz oldu. Son hata: {last_error}")
    return f"Bir hata olustu: {last_error}"


async def soru_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        await update.message.reply_text("Bu botu kullanma yetkin yok.")
        return
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Kullanim: /soru <sorunuz>")
        return
        
    eski_soru = question
    question = duzelt_metin(question)
    if eski_soru != question:
        logger.info(f"Yazim duzeltildi: '{eski_soru}' -> '{question}'")

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    loop = asyncio.get_running_loop()
    
    chat_history = update_history(user_id, "user", question)
    answer = await loop.run_in_executor(None, ask_with_fallback, chat_history)
    update_history(user_id, "assistant", answer)
    
    await update.message.reply_text(answer)


async def handle_specific_model(update: Update, context: ContextTypes.DEFAULT_TYPE, provider: str, model: str, api_key: str):
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        await update.message.reply_text("Bu botu kullanma yetkin yok.")
        return
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Kullanim: /komut <sorunuz>")
        return
    
    # Yazim hatalarini local olarak cache uzerinden duzelt
    eski_soru = question
    question = duzelt_metin(question)
    if eski_soru != question:
        logger.info(f"Yazim duzeltildi: '{eski_soru}' -> '{question}'")
        
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    if not api_key or "BURAYA" in api_key:
        await update.message.reply_text("Bu model icin API anahtari ayarlanmamis!")
        return
        
    try:
        loop = asyncio.get_running_loop()
        chat_history = update_history(user_id, "user", question)
        
        if provider == "nvidia":
            answer = await loop.run_in_executor(None, ask_nvidia, chat_history, model, api_key)
        elif provider == "deepseek":
            answer = await loop.run_in_executor(None, ask_deepseek, chat_history, model, api_key)
        else:
            answer = await loop.run_in_executor(None, ask_gemini, chat_history, model, api_key)
            
        update_history(user_id, "assistant", answer)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Hata olustu: {e}")

async def sorudeep_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_specific_model(update, context, "deepseek", DEEPSEEK_MODEL, DEEPSEEK_API_KEY)

async def sorugpt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_specific_model(update, context, "nvidia", GPT_OSS_MODEL, NVIDIA_API_KEY_1)

async def sorugemma_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_specific_model(update, context, "nvidia", "google/gemma-4-31b-it", NVIDIA_API_KEY_1)

async def sorugemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_specific_model(update, context, "gemini", GEMINI_MODEL, GEMINI_API_KEY)

async def sorumuse_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_specific_model(update, context, "nvidia", NVIDIA_MODEL_FALLBACK, NVIDIA_API_KEY_2)

async def yazim_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        await update.message.reply_text("Bu botu kullanma yetkin yok.")
        return
        
    kelime = " ".join(context.args)
    if not kelime:
        await update.message.reply_text("Kullanim: /yazim <kelime veya cumle>")
        return
        
    eski_kelime = kelime
    duzeltilmis_kelime = duzelt_metin(kelime)
    if eski_kelime != duzeltilmis_kelime:
        logger.info(f"Yazim duzeltildi: '{eski_kelime}' -> '{duzeltilmis_kelime}'")
        
    soru_formati = f"'{duzeltilmis_kelime}' nasil yazilir? TDK'ye gore dogrudan kisa ve net cevabi ver, aciklama yapma."
        
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    loop = asyncio.get_running_loop()
    
    chat_history = update_history(user_id, "user", soru_formati)
    answer = await loop.run_in_executor(None, ask_with_fallback, chat_history)
    update_history(user_id, "assistant", answer)
    
    await update.message.reply_text(answer)

async def komutlar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 **Bot Komutlari:**\n\n"
        "🔸 `/soru <soru>`\nOtomatik model secer. (Onerilen)\n"
        "🔸 `/yazim <kelime>`\nTDK'ye gore dogru yazilisini soyler (TYT).\n"
        "🔸 `/sorudeep <soru>`\nSadece DeepSeek V4.1 Flash kullanir.\n"
        "🔸 `/sorugemini <soru>`\nSadece Gemini 3.8 Flash kullanir.\n"
        "🔸 `/sorumuse <soru>`\nSadece Muse-Glimmer-30B kullanir.\n\n"
        "🔸 `/komutlar`\nBu menuyu gosterir.\n\n"
        "📸 **Fotograf Gonderimi:**\nBota dogrudan fotograf atarak soruyu cozdurebilirsin."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Merhaba! Ben Soru Botu. Istedigin soruyu sorabilirsin.\n"
        "Tum komutlari gormek icin /komutlar yazabilirsin."
    )
    await update.message.reply_text(msg)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        await update.message.reply_text("Bu botu kullanma yetkin yok.")
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        photo = update.message.photo[-1] # En yuksek cozunurluklu versiyon
        photo_file = await context.bot.get_file(photo.file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        
        caption = update.message.caption or "Bu gorseldeki soruyu coz veya ne oldugunu kisaca acikla."
        
        if not GEMINI_API_KEY or "BURAYA" in GEMINI_API_KEY:
            await update.message.reply_text("Gemini API anahtari ayarli degil! Gorselleri sadece Gemini okuyabilir.")
            return
            
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, ask_gemini_vision, caption, photo_bytes, GEMINI_API_KEY)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Gorsel islenirken hata: {e}")
        await update.message.reply_text(f"Gorsel islenirken hata olustu: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        return
    question = update.message.text
    if not question:
        return
        
    eski_soru = question
    question = duzelt_metin(question)
    if eski_soru != question:
        logger.info(f"Yazim duzeltildi: '{eski_soru}' -> '{question}'")

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    loop = asyncio.get_running_loop()
    
    chat_history = update_history(user_id, "user", question)
    answer = await loop.run_in_executor(None, ask_with_fallback, chat_history)
    update_history(user_id, "assistant", answer)
    
    await update.message.reply_text(answer)

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN tanimli degil.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("komutlar", komutlar_command))
    app.add_handler(CommandHandler("soru", soru_command))
    app.add_handler(CommandHandler("yazim", yazim_command))
    app.add_handler(CommandHandler("sorudeep", sorudeep_command))
    app.add_handler(CommandHandler("sorugemma", sorugemma_command))
    app.add_handler(CommandHandler("sorugpt", sorugpt_command))
    app.add_handler(CommandHandler("sorugemini", sorugemini_command))
    app.add_handler(CommandHandler("sorumuse", sorumuse_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot baslatildi (Multi-Model ve Gorsel Destekli).")
    
    # Render icin Flask sunucusunu arka planda baslat
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    
    # Render (Linux) ve yeni Python surumlerinde event loop hatasini onlemek icin:
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        
    app.run_polling()


if __name__ == "__main__":
    main()
