# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from bot.database.models import Whitelist
from bot.filters.sudo import SudoFilter

router = Router(name="whitelist")
router.message.filter(SudoFilter())


@router.message(Command(commands=["allow", "disallow"]))
async def change_whitelist(message: Message, command: CommandObject) -> None:
    if not message.from_user:
        return

    if not command.args:
        await message.reply("You must provide a chat ID.")
        return

    chat_id = command.args.split(" ")[0]
    try:
        chat_id_int = int(chat_id)
    except ValueError:
        await message.reply("The chat ID must be a valid number.")
        return

    if command.command == "allow":
        await Whitelist.get_or_create(chat_id=chat_id_int)
        await message.reply(f"Chat ID {chat_id} has been allowed to use the bot.")
    elif command.command == "disallow":
        await Whitelist.filter(chat_id=chat_id_int).delete()
        await message.reply(f"Chat ID {chat_id} has been disallowed to use the bot.")


@router.message(Command("list"))
async def list_whitelist(message: Message) -> None:
    if not message.from_user:
        return

    entries = await Whitelist.all().values_list("chat_id", flat=True)
    if not entries:
        await message.reply("No whitelist entries found.")
        return

    ids_list = ", ".join(str(id) for id in entries)
    await message.reply(f"Whitelisted chat IDs: {ids_list}")
