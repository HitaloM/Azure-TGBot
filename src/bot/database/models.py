# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from tortoise import fields, models


class Conversation(models.Model):
    """
    Represents a conversation between a user and the bot.

    Attributes:
        id (int): The primary key of the conversation.
        user_id (int): The unique identifier of the user.
        user_message (str): The message sent by the user.
        bot_response (str): The response generated by the bot.
        timestamp (datetime): The timestamp when the conversation occurred.
    """

    id = fields.IntField(pk=True)
    user_id = fields.BigIntField()
    user_message = fields.TextField()
    bot_response = fields.TextField()
    timestamp = fields.DatetimeField(auto_now_add=True)


class Whitelist(models.Model):
    """
    Represents a whitelist entry for allowed chats.

    Attributes:
        id (int): The primary key of the whitelist entry.
        chat_id (int): The unique identifier of the chat, must be unique.
    """

    id = fields.IntField(pk=True)
    chat_id = fields.BigIntField(unique=True)
