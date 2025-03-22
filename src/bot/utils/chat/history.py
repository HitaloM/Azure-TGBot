# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from azure.ai.inference.models import AssistantMessage, UserMessage

from bot.database.models import Conversation


async def get_conversation_history(user_id: int) -> list[AssistantMessage | UserMessage]:
    """
    Retrieve the conversation history for a specific user.

    This asynchronous function fetches the conversation records associated with the given
    user, orders them by timestamp, and returns an interleaved list of user and assistant messages.

    Args:
        user_id (int): The unique identifier of the user.

    Returns:
        list[AssistantMessage | UserMessage]: A list of messages in the conversation history.
    """
    records = await Conversation.filter(user_id=user_id).order_by("-timestamp").limit(30).all()
    records.reverse()

    return [
        msg
        for record in records
        for msg in (
            UserMessage(content=record.user_message),
            AssistantMessage(content=record.bot_response),
        )
    ]


async def clear_conversation_history(user_id: int) -> None:
    """
    Clears the conversation history for a specific user.

    This function deletes all conversation records associated with the given user ID.

    Args:
        user_id (int): The unique identifier of the user whose conversation history
            should be cleared.
    """
    await Conversation.filter(user_id=user_id).delete()
