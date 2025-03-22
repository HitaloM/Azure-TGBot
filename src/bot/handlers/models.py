# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.utils.chat.models import AIModel

router = Router(name="models")


@router.message(Command("models"))
async def list_models_handler(message: Message) -> None:
    models_html = "\n".join(f"• <code>{model}</code>" for model in AIModel.list_models())
    response = f"<b>Supported Models:</b>\n{models_html}"
    await message.answer(response, parse_mode="HTML")
