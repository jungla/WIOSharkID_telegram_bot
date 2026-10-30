import io
import os
from pathlib import Path
from PIL import Image
import logging
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Import your ONNX classifier engine

# Import the classifier engine
from standalone_engine import StandaloneElasmobranchClassifier

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/elasmobranch_mobilenet_int8.onnx")
META_PATH = os.getenv("META_PATH", "./models/metadata.json")

import asyncio
import io
import logging
import os
from typing import Dict, List
from PIL import Image
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

# Import the standalone classifier engine
from standalone_engine import StandaloneElasmobranchClassifier

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION & MODEL INITIALIZATION
# ==========================================

# Initialize classifier on startup
logger.info(f"Loading ONNX model: {MODEL_PATH}")
classifier = StandaloneElasmobranchClassifier(
    onnx_model_path=MODEL_PATH,
    meta_path=META_PATH
)
logger.info("Model loaded successfully into memory.")

# Global dictionary to accumulate multi-photo albums (media groups)
# Format: { media_group_id: [ (file_id, update_object), ... ] }
MEDIA_GROUPS: Dict[str, List] = {}
MEDIA_GROUP_LOCK = asyncio.Lock()

def render_ascii_bar(pct: float, width: int = 10) -> str:
    """Render a simple ASCII confidence bar for text output."""
    filled = int(round(pct * width))
    empty = width - filled
    return "█" * filled + "░" * empty

# ==========================================
# CORE INFERENCE PROCESSOR
# ==========================================
async def process_and_reply_batch(updates: List[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads all gathered images for a single message/album and runs batched inference."""
    first_update = updates[0]
    chat_id = first_update.effective_chat.id

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        pil_images: List[Image.Image] = []

        # Download all photos in the batch
        for upd in updates:
            photo_file = await upd.message.photo[-1].get_file(read_timeout=30)
            photo_bytes = await photo_file.download_as_bytearray()
            img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
            pil_images.append(img)

        logger.info(f"Processing batch of {len(pil_images)} photo(s) for chat {chat_id}")

        # Offload multi-photo batch inference to a background thread to prevent blocking asyncio loop
        result = await asyncio.to_thread(classifier.predict, pil_images)

        top_sp = result["top_species"]
        top_conf = result["top_confidence"]
        tier = result["confidence_tier"]
        num_photos = result["num_photos_analyzed"]
        rec = result["recommendation"]

        # Format Markdown output response
        lines = []
        lines.append(f"🔍 *Primary Result:* _{top_sp}_")
        lines.append(f"📊 *Confidence:* `{top_conf * 100:.1f}%` ({tier})")
        lines.append(f"📸 *Photos Analyzed:* `{num_photos}` photo(s) pooled\n")

        lines.append("🏆 *Top Candidates:*")
        for rank, cand in enumerate(result["top_k"], 1):
            bar = render_ascii_bar(cand["confidence"])
            lines.append(f"{rank}. _{cand['species']}_ — `{cand['confidence_pct']}` {bar}")

        lines.append("\n📌 *Diagnostic Guidance:*")
        lines.append(f"{rec['message']}")

        if rec.get("diagnostic_notes"):
            lines.append(f"\n🧬 *Morphological Key:*\n_{rec['diagnostic_notes']}_")

        if rec.get("suggested_angles"):
            lines.append("\n📸 *Recommended Photo Angles for Confirmation:*")
            for angle in rec["suggested_angles"]:
                lines.append(f"• {angle}")

        response_text = "\n".join(lines)
        await first_update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error processing image batch: {e}", exc_info=True)
        await first_update.message.reply_text(
            f"❌ Error processing image batch: `{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )

# ==========================================
# ASYNC HANDLERS
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when /start or /help is issued."""
    welcome_text = (
        "🦈 *WIO Sharks & Rays Species ID Bot*\n\n"
        "Send one or multiple photos (as an album) of an elasmobranch specimen "
        "(shark, ray, or guitarfish) to run automated species identification."
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles single photos and media group (album) photo messages.
    Buffers multi-photo uploads before calling batch prediction.
    """
    media_group_id = update.message.media_group_id

    # If it's a standalone single photo (not part of an album)
    if not media_group_id:
        await process_and_reply_batch([update], context)
        return

    # If it's part of a multi-photo album dispatch
    async with MEDIA_GROUP_LOCK:
        if media_group_id not in MEDIA_GROUPS:
            MEDIA_GROUPS[media_group_id] = []
            
            # Schedule execution after 1.5s buffer to allow Telegram to deliver all album photos
            asyncio.create_task(accumulate_and_process_group(media_group_id, context))

        MEDIA_GROUPS[media_group_id].append(update)

async def accumulate_and_process_group(media_group_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Waits for album updates to finish streaming, then triggers batch inference."""
    await asyncio.sleep(1.5)  # Wait for remaining photos in album to arrive

    async with MEDIA_GROUP_LOCK:
        updates = MEDIA_GROUPS.pop(media_group_id, [])

    if updates:
        await process_and_reply_batch(updates, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback handler for non-photo messages."""
    await update.message.reply_text(
        "Please upload or send a photo attachment (or multi-photo album) to run species identification."
    )

# ==========================================
# APPLICATION BUILDER & RUNNER
# ==========================================
def main() -> None:
    """Configures HTTP timeouts, attaches handlers, and launches polling."""
    # Custom HTTP timeouts to prevent TimedOut errors during image transfers
    request_config = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=20.0
    )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request_config)
        .build()
    )

    # Register Handlers
    app.add_handler(CommandHandler(["start", "help"], start_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is polling for incoming messages...")
    app.run_polling()

if __name__ == "__main__":
    main()
