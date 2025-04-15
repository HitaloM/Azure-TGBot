# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from sqlalchemy import delete, select

from bot.database.connection import async_session
from bot.database.models import Conversation, Whitelist


async def get_whitelist_entry(chat_id: int) -> Whitelist | None:
    """
    Get a whitelist entry by chat_id.

    Args:
        chat_id: The chat ID to search for

    Returns:
        The whitelist entry if found, None otherwise
    """
    async with async_session() as session:
        stmt = select(Whitelist).where(Whitelist.chat_id == chat_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def add_to_whitelist(chat_id: int) -> Whitelist:
    """
    Add a chat ID to the whitelist.

    Args:
        chat_id: The chat ID to add to the whitelist

    Returns:
        The created whitelist entry
    """
    async with async_session() as session:
        stmt = select(Whitelist).where(Whitelist.chat_id == chat_id)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            return existing

    async with async_session() as session, session.begin():
        whitelist_entry = Whitelist(chat_id=chat_id)
        session.add(whitelist_entry)
        await session.flush()
        return whitelist_entry


async def remove_from_whitelist(chat_id: int) -> bool:
    """
    Remove a chat ID from the whitelist.

    Args:
        chat_id: The chat ID to remove from the whitelist

    Returns:
        True if the entry was removed, False if it didn't exist
    """
    async with async_session() as session, session.begin():
        stmt = delete(Whitelist).where(Whitelist.chat_id == chat_id)
        result = await session.execute(stmt)
        return result.rowcount > 0


async def get_all_whitelist_entries() -> list[int]:
    """
    Get all chat IDs from the whitelist.

    Returns:
        List of all whitelisted chat IDs
    """
    async with async_session() as session:
        stmt = select(Whitelist.chat_id)
        result = await session.execute(stmt)
        return [row[0] for row in result.all()]


async def save_conversation(user_id: int, user_message: str, bot_response: str) -> Conversation:
    """
    Save a conversation exchange to the database.

    Args:
        user_id: The user ID
        user_message: The user's message
        bot_response: The bot's response

    Returns:
        The created conversation entry
    """
    async with async_session() as session, session.begin():
        conversation = Conversation(
            user_id=user_id, user_message=user_message, bot_response=bot_response
        )
        session.add(conversation)
        return conversation


async def prune_conversation_history(user_id: int, keep_count: int = 30) -> int:
    """
    Prune conversation history to keep only the latest entries.

    Args:
        user_id: The user ID
        keep_count: Number of latest messages to keep

    Returns:
        Number of pruned records
    """
    async with async_session() as session:
        subq = (
            select(Conversation.id)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.timestamp.desc())
            .limit(keep_count)
            .scalar_subquery()
        )

        async with session.begin():
            stmt = delete(Conversation).where(
                Conversation.user_id == user_id, Conversation.id.not_in(subq)
            )
            result = await session.execute(stmt)
            return result.rowcount


async def get_user_conversation_history(user_id: int, limit: int = 30) -> list[Conversation]:
    """
    Get conversation history for a user.

    Args:
        user_id: The user ID
        limit: Maximum number of conversations to retrieve

    Returns:
        List of conversation entries, oldest first
    """
    async with async_session() as session:
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
        return list(reversed(records))


async def clear_user_conversation_history(user_id: int) -> int:
    """
    Clear conversation history for a user.

    Args:
        user_id: The user ID

    Returns:
        Number of rows deleted
    """
    async with async_session() as session, session.begin():
        stmt = delete(Conversation).where(Conversation.user_id == user_id)
        result = await session.execute(stmt)
        return result.rowcount


async def clear_all_conversation_history() -> int:
    """
    Clear conversation history for all users.

    Returns:
        Number of rows deleted
    """
    async with async_session() as session, session.begin():
        stmt = delete(Conversation)
        result = await session.execute(stmt)
        return result.rowcount
