# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram.filters import BaseFilter
from aiogram.types import Message

from bot import config


class SudoFilter(BaseFilter):
    """
    A filter to check if a user is a sudoer.

    Methods:
        __call__(message: Message) -> bool:
            Determines whether the user who sent the message is in the sudoers list.
    """

    @staticmethod
    async def __call__(message: Message) -> bool:
        """
        Checks if the user who sent the message is in the sudoers list.

        Args:
            message (Message): The incoming message to be filtered.

        Returns:
            bool: True if the user is a sudoer, False otherwise.
        """
        if not message.from_user:
            return False

        return message.from_user.id in config.sudoers
