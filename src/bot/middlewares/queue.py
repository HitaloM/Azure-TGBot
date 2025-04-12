# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)


class QueueMiddleware(BaseMiddleware):
    """
    Middleware to handle message processing in a queue per user per chat.

    This middleware ensures that messages from the same user in the same chat are processed
    sequentially by maintaining a queue for each user in each chat.

    Attributes:
        user_queues (dict[str, asyncio.Queue]): A dictionary mapping user-chat IDs to their
            respective queues.
        tasks (dict[str, asyncio.Task]): A dictionary mapping user-chat IDs to their respective
            queue processor tasks.
        max_queue_size (int): Maximum number of messages in the queue before rejecting new ones.
        process_timeout (float): Maximum time in seconds to process a message before timing out.
        cleanup_interval (int): Time in seconds between queue cleanup checks.
        idle_timeout (int): Time in seconds after which an idle queue is removed.
        last_activity (dict[str, float]): Tracks the last activity time for each user-chat.
    """

    def __init__(
        self,
        max_queue_size: int = 50,
        process_timeout: float = 60.0,
        cleanup_interval: int = 300,
        idle_timeout: int = 3600,
    ):
        """
        Initialize the queue middleware with configurable parameters.

        Args:
            max_queue_size (int): Maximum size of each user's queue. Defaults to 50.
            process_timeout (float): Maximum time in seconds to process a message. Defaults to 60.
            cleanup_interval (int): Time between cleanup checks in seconds. Defaults to 300
                (5 minutes).
            idle_timeout (int): Time after which an idle queue is removed in seconds. Defaults to
                3600 (1 hour).
        """
        self.user_queues: dict[str, asyncio.Queue] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.max_queue_size = max_queue_size
        self.process_timeout = process_timeout
        self.cleanup_interval = cleanup_interval
        self.idle_timeout = idle_timeout
        self.last_activity: dict[str, float] = {}

        # Start the cleanup task
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.debug(
            "Queue middleware initialized with max_queue_size=%d, process_timeout=%.1f",
            max_queue_size,
            process_timeout,
        )

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        """
        Enqueues a message for processing and ensures a queue exists for the user in the chat.

        Args:
            handler: The handler function to process the message.
            event: The incoming message event.
            data: Additional data for the handler.

        Returns:
            Any: The result of the handler if immediate processing is enabled.
        """
        # Only process Message objects
        if not isinstance(event, Message):
            return await handler(event, data)

        # Skip if no user - should never happen but just in case
        if not event.from_user:
            return await handler(event, data)

        chat_id = str(event.chat.id)
        user_id = str(event.from_user.id)
        queue_id = f"{user_id}:{chat_id}"

        # Update last activity time
        self.last_activity[queue_id] = time.time()

        # Create a new queue if it doesn't exist
        self._ensure_queue(queue_id)

        queue = self.user_queues[queue_id]

        # Check if queue is full
        if queue.qsize() >= self.max_queue_size:
            logger.warning(
                "Queue for user %s in chat %s is full. Message rejected.", user_id, chat_id
            )
            return None

        # Put the message in the queue with its timestamp
        item = (time.time(), (handler, event, data))
        try:
            await asyncio.wait_for(queue.put(item), timeout=5.0)
            logger.debug("Message enqueued for user %s in chat %s", user_id, chat_id)
        except TimeoutError:
            logger.error(
                "Timeout while trying to enqueue message for user %s in chat %s", user_id, chat_id
            )
            return None

        # Return None as the actual processing will happen in the queue
        return None

    def _ensure_queue(self, queue_id: str) -> None:
        """
        Ensures that a queue and a processing task exist for the given queue ID.

        Args:
            queue_id: The ID of the queue (user_id:chat_id).
        """
        if queue_id not in self.user_queues:
            self.user_queues[queue_id] = asyncio.Queue()
            self.tasks[queue_id] = asyncio.create_task(self.queue_processor(queue_id))
            self.last_activity[queue_id] = time.time()
            logger.debug("Created new queue and task for queue_id: %s", queue_id)

    async def queue_processor(self, queue_id: str) -> None:
        """
        Processes messages in the queue for a specific user in a specific chat.

        Args:
            queue_id: The ID of the queue (user_id:chat_id).
        """
        queue = self.user_queues[queue_id]
        while True:
            try:
                # Get the next message from the queue
                _timestamp, (handler, event, data) = await queue.get()

                # Update activity timestamp
                self.last_activity[queue_id] = time.time()

                # Process the message with timeout
                try:
                    logger.debug("Processing message for queue_id: %s", queue_id)
                    await asyncio.wait_for(handler(event, data), timeout=self.process_timeout)
                    logger.debug("Message processed for queue_id: %s", queue_id)
                except TimeoutError:
                    logger.warning("Message processing timed out for queue_id: %s", queue_id)
                except Exception as e:
                    logger.exception("Error processing message for queue_id: %s: %s", queue_id, e)
                finally:
                    queue.task_done()
                    logger.debug(
                        "Task done for queue_id: %s, queue size: %d", queue_id, queue.qsize()
                    )
            except asyncio.CancelledError:
                logger.info("Queue processor for queue_id: %s cancelled", queue_id)
                break
            except Exception as e:
                logger.exception(
                    "Unexpected error in queue processor for queue_id: %s: %s", queue_id, e
                )
                # Sleep a bit to avoid tight loop in case of persistent errors
                await asyncio.sleep(1)

    async def _periodic_cleanup(self) -> None:
        """
        Periodically checks for and removes idle queues to free up resources.
        """
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_queues()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error during queue cleanup: %s", e)

    async def _cleanup_idle_queues(self) -> None:
        """
        Removes queues that have been idle for longer than the idle_timeout.
        """
        current_time = time.time()
        queue_ids_to_remove = []

        for queue_id, last_active in self.last_activity.items():
            if current_time - last_active > self.idle_timeout:
                queue_ids_to_remove.append(queue_id)

        for queue_id in queue_ids_to_remove:
            await self._remove_queue(queue_id)

        if queue_ids_to_remove:
            logger.info("Cleaned up %d idle queues", len(queue_ids_to_remove))

    async def _remove_queue(self, queue_id: str) -> None:
        """
        Safely removes a queue and cancels its processor task.

        Args:
            queue_id: The ID of the queue to remove (user_id:chat_id).
        """
        if queue_id in self.tasks:
            task = self.tasks[queue_id]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            del self.tasks[queue_id]

        if queue_id in self.user_queues:
            del self.user_queues[queue_id]

        if queue_id in self.last_activity:
            del self.last_activity[queue_id]

        logger.debug("Removed queue and task for queue_id: %s", queue_id)

    async def _shutdown(self) -> None:
        """
        Properly shuts down all queues and tasks when the application is stopping.
        Should be called during application shutdown.

        Note: This is not part of the BaseMiddleware interface, but used by our middleware manager.
        """
        logger.info("Shutting down queue middleware...")

        # Cancel the cleanup task
        if hasattr(self, "cleanup_task"):
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task

        # Cancel all queue processor tasks
        for _queue_id, task in list(self.tasks.items()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.tasks.clear()
        self.user_queues.clear()
        self.last_activity.clear()

        logger.info("Queue middleware shutdown complete")
