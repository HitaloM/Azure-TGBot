# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram.filters import BaseFilter
from aiogram.types import Message

from bot import config


class SudoFilter(BaseFilter):
    """
    Filter to check if a user has sudo privileges.

    This filter checks whether the user who sent the message is in the sudoers list
    defined in the application configuration. Users in this list are granted
    administrative privileges.
    """

    @staticmethod
    async def __call__(message: Message) -> bool:
        """
        Check if the message sender has sudo privileges.

        This method verifies if the user who sent the message is present in the
        application's sudoers list. If there's no user associated with the message,
        it returns False.

        Args:
            message (Message): The incoming Telegram message to be filtered.

        Returns:
            bool: True if the user is in the sudoers list, False otherwise.
        """
        if not message.from_user:
            return False

        return message.from_user.id in config.sudoers
