# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from azure.ai.inference.models import AssistantMessage, UserMessage

from bot.database import get_user_conversation_history

type ChatHistory = list[AssistantMessage | UserMessage]


async def get_conversation_history(user_id: int, chat_id: int) -> ChatHistory:
    """
    Retrieve conversation history for a user in a specific chat.

    Fetches up to 30 most recent conversation records for the user in the specified chat,
    orders them chronologically, and converts them to interleaved UserMessage and
    AssistantMessage objects.

    Args:
        user_id: Unique identifier for the user
        chat_id: Unique identifier for the chat (group or private)

    Returns:
        List of alternating UserMessage and AssistantMessage objects
    """
    records = await get_user_conversation_history(user_id, chat_id)

    messages: ChatHistory = []
    for record in records:
        messages.append(UserMessage(content=record.user_message))
        messages.append(AssistantMessage(content=record.bot_response))

    return messages
