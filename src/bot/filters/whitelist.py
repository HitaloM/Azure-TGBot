# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram.enums import ChatType
from aiogram.filters import BaseFilter
from aiogram.types import Message

from bot import config
from bot.database.models import Whitelist


class WhiteListFilter(BaseFilter):
    """
    A filter to check if a user or chat is whitelisted.

    Methods:
        __call__(message: Message) -> bool:
            Determines whether the user or chat is in the whitelist or is a sudoer.
    """

    @staticmethod
    async def __call__(message: Message) -> bool:
        """
        Checks if the user or chat is whitelisted or if the user is a sudoer.

        Args:
            message (Message): The incoming message to be filtered.

        Returns:
            bool: True if the user or chat is whitelisted or the user is a sudoer, False otherwise.
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
        is_whitelisted = await Whitelist.get_or_none(chat_id=chat_id)

        return bool(is_whitelisted)
