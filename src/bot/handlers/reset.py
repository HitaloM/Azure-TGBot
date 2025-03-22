# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.utils.chat.history import clear_conversation_history

router = Router(name="reset")


@router.message(Command(commands=["reset", "fuck"]))
async def reset_handler(message: Message) -> None:
    if not message.from_user:
        return

    await clear_conversation_history(message.from_user.id)
    await message.answer("History cleared.")
