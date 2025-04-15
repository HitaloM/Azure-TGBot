# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram.enums import ChatType
from aiogram.filters import BaseFilter
from aiogram.types import Message

from bot import config
from bot.database import get_whitelist_entry


class WhiteListFilter(BaseFilter):
    """
    Filter to check if a user or chat is whitelisted.

    This filter verifies if a chat or user is authorized to use the bot based on:
    1. If the user is in the sudo list (admins always have access)
    2. If the chat ID exists in the whitelist database table

    For group chats, the chat ID is checked against the whitelist.
    For private chats, the user ID is checked against the whitelist.
    """

    @staticmethod
    async def __call__(message: Message) -> bool:
        """
        Check if the user or chat is authorized to use the bot.

        This method verifies if:
        - The user is a sudoer (admin)
        - The chat ID is in the whitelist database

        For group chats, it checks the chat ID against the whitelist.
        For private chats, it checks the user ID against the whitelist.

        Args:
            message (Message): The incoming Telegram message to be filtered.

        Returns:
            bool: True if the user/chat is authorized to use the bot, False otherwise.
        """
        if not message.from_user:
            return False

        if message.from_user.id in config.sudoers:
            return True

        chat_id = (
            message.chat.id
            if message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
            else message.from_user.id
        )

        is_whitelisted = await get_whitelist_entry(chat_id)
        return bool(is_whitelisted)
