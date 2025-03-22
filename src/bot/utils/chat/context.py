# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram.types import Message
from azure.ai.inference.models import AssistantMessage, UserMessage


def build_reply_context(
    message: Message, user_prompt: str, history: list[AssistantMessage | UserMessage]
) -> tuple[str, list[AssistantMessage | UserMessage]]:
    """
    Build the reply context by incorporating the replied message content into the user prompt
    and updating the conversation history.

    Args:
        message (Message): The message that may contain a reply.
        user_prompt (str): The user's input prompt.
        history (list[AssistantMessage | UserMessage]): The conversation history.

    Returns:
        tuple[str, list[AssistantMessage | UserMessage]]: A tuple containing the updated prompt
        and the updated conversation history.
    """
    reply_msg = message.reply_to_message

    if not reply_msg or not (reply_msg.text or reply_msg.caption) or not reply_msg.from_user:
        return user_prompt.strip(), history

    content = (reply_msg.text or reply_msg.caption).strip()  # type: ignore
    sender = reply_msg.from_user
    sender_name = sender.full_name.strip()

    if message.bot and sender.id == message.bot.id:
        note = "User replied to a bot message."
    elif message.from_user and sender.id == message.from_user.id:
        note = "User replied to their own message."
    else:
        note = "User replied to another user's message."

    context_lines = [
        f"_instruction: {note} Use the replied message as context.",
        f"Replied message content: '{content}'",
    ]
    if note == "User replied to another user's message.":
        context_lines.append(f"Sent by: {sender_name}")

    context = "\n".join(context_lines)
    updated_prompt = f"{context}\nUser Prompt: {user_prompt.strip()}"

    history.append(UserMessage(content=updated_prompt))
    return updated_prompt, history
