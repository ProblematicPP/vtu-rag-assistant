"""Optional Telegram bot: `docker compose --profile telegram up -d telegram-bot`.

Commands:
    /start              intro
    /subject BCS303     restrict answers to one subject (/subject all to clear)
    /subjects           list subjects for the default branch/scheme
    <any text>          ask a question (agentic mode)
"""

import logging
from typing import Any

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from vtu_rag.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
API = settings.api_base_url.rstrip("/")
TELEGRAM_LIMIT = 4000  # Telegram caps messages at 4096 chars


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I'm ChatVTU. Ask me anything from your "
        f"{settings.default_branch.upper()} {settings.default_scheme}-scheme notes.\n\n"
        "Use /subject BCS303 to focus on one subject, /subjects to list them."
    )


async def list_subjects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"{API}/api/v1/subjects",
            params={"branch": settings.default_branch, "scheme": settings.default_scheme},
        )
    if response.is_error:
        await update.message.reply_text("Couldn't load subjects right now.")
        return
    lines = [f"Sem {s['semester']} · {s['code']} — {s['name']}" for s in response.json()]
    await update.message.reply_text("\n".join(lines) or "No subjects yet.")


async def set_subject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        current = context.user_data.get("subject_code", "all subjects")
        await update.message.reply_text(f"Currently searching: {current}. Usage: /subject BCS303")
        return
    code = context.args[0].upper()
    if code == "ALL":
        context.user_data.pop("subject_code", None)
        await update.message.reply_text("Searching all subjects.")
    else:
        context.user_data["subject_code"] = code
        await update.message.reply_text(f"Answers will now come from {code} notes.")


def _format(data: dict[str, Any]) -> str:
    text = data["answer"]
    cited = [s for s in data.get("sources", []) if s.get("cited")] or data.get("sources", [])
    if cited:
        text += "\n\nSources:"
        for s in cited:
            section = (s.get("section_heading") or "").split(" > ")[-1]
            text += (
                f"\n[{s['index']}] {s['subject_code']} M{s['module_number']}"
                f"{' · ' + section if section else ''} ({s['note_title']})"
            )
    return text[:TELEGRAM_LIMIT]


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = (update.message.text or "").strip()
    if len(question) < 3:
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    payload: dict[str, Any] = {"question": question}
    if code := context.user_data.get("subject_code"):
        payload["subject_code"] = code
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(f"{API}/api/v1/agentic-ask", json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("API call failed: %s", exc)
        await update.message.reply_text("Sorry, I can't reach the study assistant right now.")
        return
    await update.message.reply_text(_format(response.json()))


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("subjects", list_subjects))
    app.add_handler(CommandHandler("subject", set_subject))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))
    app.run_polling()


if __name__ == "__main__":
    main()
