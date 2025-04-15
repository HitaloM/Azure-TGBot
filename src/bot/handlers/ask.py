# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import Message

from bot.utils.chat.response_processor import process_and_reply

router = Router(name="ask")


@router.message(F.chat.type == ChatType.PRIVATE)
async def pm_message_handler(message: Message) -> None:
    await process_and_reply(message)


@router.message(
    Command(commands=["ask", "ai"]),
    F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
)
async def group_command_handler(message: Message) -> None:
    await process_and_reply(message, clear=True)


@router.message(
    F.reply_to_message.from_user.id == F.bot.id,
    F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}),
)
async def group_reply_handler(message: Message) -> None:
    await process_and_reply(message)
