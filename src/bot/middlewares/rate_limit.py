# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.types import CallbackQuery, Message, TelegramObject

from bot import config

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseMiddleware):
    """Middleware to control and limit request rates for users and groups.

    This middleware prevents abuse and overload by limiting how many requests a user or group
    can make in a specific time period. Administrators defined in config.sudoers are exempt
    from rate limiting.

    Attributes:
        user_requests: Dictionary that stores timestamps of requests for each user.
        group_requests: Dictionary that stores timestamps of requests for each group.
        user_limit: Maximum number of requests allowed per user in the interval.
        user_interval: Time interval (in seconds) to check user requests.
        group_limit: Maximum number of requests allowed per group in the interval.
        group_interval: Time interval (in seconds) to check group requests.
        cleanup_task: Asyncio task that runs periodic cleanup of old records.
    """

    def __init__(
        self,
        user_limit: int = 5,
        user_interval: int = 30,
        group_limit: int = 10,
        group_interval: int = 60,
        cleanup_interval: int = 300,
    ):
        """Initialize the rate limiting middleware with configurable parameters.

        Args:
            user_limit: Maximum number of requests per user in the interval. Defaults to 5.
            user_interval: Time interval in seconds for user rate limiting. Defaults to 30.
            group_limit: Maximum number of requests per group in the interval. Defaults to 10.
            group_interval: Time interval in seconds for group rate limiting. Defaults to 60.
            cleanup_interval: Interval in seconds to clean old records.
                Defaults to 300 (5 minutes).
        """
        # Using defaultdict to automatically create empty lists for new users/groups
        self.user_requests: dict[int, list[float]] = defaultdict(list)
        self.group_requests: dict[int, list[float]] = defaultdict(list)

        self.user_limit = user_limit
        self.user_interval = user_interval
        self.group_limit = group_limit
        self.group_interval = group_interval

        # Start periodic cleanup task
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup(cleanup_interval))
        logger.debug(
            "Rate limit middleware initialized: user_limit=%d, "
            "user_interval=%d, group_limit=%d, group_interval=%d",
            user_limit,
            user_interval,
            group_limit,
            group_interval,
        )

    async def __call__(  # noqa: C901
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: Message | CallbackQuery,
        data: dict[str, Any],
    ) -> Any | None:
        """Process an event and apply rate limiting.

        This method is called by the aiogram framework for each incoming event.
        It identifies the user and/or chat, checks if they've exceeded their rate limits,
        and either allows the event to be processed or rejects it.

        Args:
            handler: The handler function that will process the event.
            event: The Telegram event (Message or CallbackQuery).
            data: Additional data for the handler.

        Returns:
            The handler result if rate limit is not exceeded, None otherwise.
        """
        now = time.time()

        # Identify user
        user_id = None
        if hasattr(event, "from_user") and event.from_user:
            user_id = event.from_user.id

        # Identify chat (group or private)
        chat_id = None
        is_group = False

        if isinstance(event, Message) and event.chat:
            chat_id = event.chat.id
            is_group = event.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
        elif isinstance(event, CallbackQuery) and event.message and event.message.chat:
            chat_id = event.message.chat.id
            is_group = event.message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}

        # If we can't identify either user or chat, allow the request
        if user_id is None and chat_id is None:
            logger.warning("Could not identify user or chat in request")
            return await handler(event, data)

        # Check limits for user
        if user_id is not None and not self._is_admin(user_id):
            if self._is_rate_limited(
                self.user_requests[user_id], self.user_limit, self.user_interval
            ):
                logger.warning("User %d exceeded the request limit", user_id)
                # Here you can send a message to the user informing about the rate limit
                if isinstance(event, Message) and event.chat.type == "private":
                    try:
                        await event.answer(
                            "⚠️ You are sending commands too quickly! "
                            f"Please wait {self.user_interval} more seconds."
                        )
                    except Exception as e:
                        logger.error("Error sending rate limit warning: %s", e)
                return None

            # Register the user's request
            self.user_requests[user_id].append(now)

        # Check limits for group
        if is_group and chat_id is not None:
            if self._is_rate_limited(
                self.group_requests[chat_id], self.group_limit, self.group_interval
            ):
                logger.warning("Group %d exceeded the request limit", chat_id)
                return None

            # Register the group's request
            self.group_requests[chat_id].append(now)

        # If passing checks, process normally
        return await handler(event, data)

    @staticmethod
    def _is_admin(user_id: int) -> bool:
        """Check if a user is a bot administrator (bypass rate limit).

        Administrators defined in the config.sudoers list are exempt from rate limits.

        Args:
            user_id: User ID to check.

        Returns:
            True if the user is an admin (in sudoers list), False otherwise.
        """
        return user_id in config.sudoers

    @staticmethod
    def _is_rate_limited(request_times: list[float], limit: int, interval: int) -> bool:
        """Check if a set of requests exceeds the rate limit.

        This method updates the request_times list in-place, removing outdated entries
        and keeping only those within the specified time interval.

        Args:
            request_times: List of request timestamps.
            limit: Maximum number of allowed requests.
            interval: Time interval in seconds.

        Returns:
            True if the limit is exceeded, False otherwise.
        """
        if not request_times:
            return False

        now = time.time()

        # Filter requests within the interval
        recent_requests = [t for t in request_times if now - t <= interval]

        # Update the list to keep only recent requests
        request_times.clear()
        request_times.extend(recent_requests)

        # Check if exceeds the limit
        return len(recent_requests) >= limit

    async def _periodic_cleanup(self, interval: int) -> None:
        """Periodically clean old request records to prevent memory growth.

        This internal coroutine runs at regular intervals, removing outdated
        request records that are no longer needed for rate limiting calculations.

        Args:
            interval: Interval between cleanups in seconds.
        """
        while True:
            try:
                await asyncio.sleep(interval)
                await self._cleanup_old_requests()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error during cleanup of old records: %s", e)

    async def _cleanup_old_requests(self) -> None:
        """Remove old request records that are no longer needed for rate limiting.

        This method checks all user and group request records and removes those
        that are outside the relevant time intervals, as well as empty entries.
        """
        now = time.time()
        user_count = 0
        group_count = 0

        # Clean user records
        for user_id, timestamps in list(self.user_requests.items()):
            # Keep only requests within the interval
            recent = [t for t in timestamps if now - t <= self.user_interval]
            if recent:
                self.user_requests[user_id] = recent
            else:
                del self.user_requests[user_id]
                user_count += 1

        # Clean group records
        for group_id, timestamps in list(self.group_requests.items()):
            # Keep only requests within the interval
            recent = [t for t in timestamps if now - t <= self.group_interval]
            if recent:
                self.group_requests[group_id] = recent
            else:
                del self.group_requests[group_id]
                group_count += 1

        if user_count > 0 or group_count > 0:
            logger.debug(
                "Cleanup completed: %d users and %d groups removed", user_count, group_count
            )

    async def _shutdown(self) -> None:
        """Properly shut down the middleware when the application is stopping.

        Cancels the cleanup task and clears all data structures to ensure a clean shutdown.

        Note:
            This is not part of the BaseMiddleware interface, but is used by the
            middleware manager during application shutdown.
        """
        logger.info("Shutting down rate limit middleware...")

        if hasattr(self, "cleanup_task"):
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task

        self.user_requests.clear()
        self.group_requests.clear()

        logger.info("Rate limit middleware shut down successfully")
