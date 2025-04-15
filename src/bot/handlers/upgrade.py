# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import os
import subprocess
import sys

from aiogram import Router
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import CallbackQuery, InaccessibleMessage, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.filters.sudo import SudoFilter

router = Router(name="upgrade")
router.message.filter(SudoFilter())


class UpgradeCallbackFactory(CallbackData, prefix="upgrade"):
    action: str


def get_git_commits() -> str:
    subprocess.run(["git", "fetch"], check=False)
    result = subprocess.run(
        ["git", "log", "HEAD..origin/main", "--oneline"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


@router.message(Command(commands=["upgrade"]))
async def upgrade_handler(message: Message) -> None:
    if not message.from_user:
        return

    commits = get_git_commits()
    if not commits:
        await message.answer("No updates available.")
        return

    kb = InlineKeyboardBuilder()
    kb.button(text="Upgrade & Restart", callback_data=UpgradeCallbackFactory(action="confirm"))

    await message.answer(
        f"New commits available:\n<pre>{commits}</pre>", reply_markup=kb.as_markup()
    )


@router.callback_query(UpgradeCallbackFactory(action="confirm").filter())
async def upgrade_confirm_callback(call: CallbackQuery) -> None:
    msg = call.message
    if isinstance(msg, InaccessibleMessage) or msg is None:
        return

    await msg.edit_text("Upgrading... Please wait.")

    proc = await asyncio.create_subprocess_exec(
        "git",
        "pull",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        await msg.answer(f"Upgrade failed:\n<pre>{stderr.decode()}</pre>")
        return

    await msg.answer("Upgrade successful. Restarting bot...")
    await asyncio.sleep(1)

    python = sys.executable
    os.execl(python, python, *sys.argv)
