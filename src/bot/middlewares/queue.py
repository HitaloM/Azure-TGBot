# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)


class QueueMiddleware(BaseMiddleware):
    """
    Middleware to handle message processing in a queue per chat.

    This middleware ensures that messages from the same chat are processed sequentially
    by maintaining a queue for each chat.

    Attributes:
        chat_queues (dict[str, asyncio.Queue]): A dictionary mapping chat IDs to their respective
            queues.
        tasks (dict[str, asyncio.Task]): A dictionary mapping chat IDs to their respective queue
            processor tasks.
    """

    def __init__(self):
        super().__init__()
        self.chat_queues: dict[str, asyncio.Queue] = {}
        self.tasks: dict[str, asyncio.Task] = {}

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ):
        """
        Enqueues a message for processing and ensures a queue exists for the chat.

        Args:
            handler (Callable): The handler function to process the message.
            event (Message): The incoming message event.
            data (dict): Additional data for the handler.
        """
        chat_id = str(event.chat.id)
        self._ensure_queue(chat_id)
        await self.chat_queues[chat_id].put((handler, event, data))
        logger.debug("Message enqueued for chat_id: %s", chat_id)

    def _ensure_queue(self, chat_id: str) -> None:
        """
        Ensures that a queue and a processing task exist for the given chat ID.

        Args:
            chat_id (str): The ID of the chat.
        """
        if chat_id not in self.chat_queues:
            self.chat_queues[chat_id] = asyncio.Queue()
            self.tasks[chat_id] = asyncio.create_task(self.queue_processor(chat_id))
            logger.debug("Created new queue and task for chat_id: %s", chat_id)

    async def queue_processor(self, chat_id: str):
        """
        Processes messages in the queue for a specific chat.

        Args:
            chat_id (str): The ID of the chat whose queue is being processed.
        """
        queue = self.chat_queues[chat_id]
        while True:
            handler, event, data = await queue.get()
            try:
                logger.debug("Processing message for chat_id: %s", chat_id)
                await handler(event, data)
                logger.debug("Message processed for chat_id: %s", chat_id)
            except Exception as e:
                logger.exception("Error processing message for chat_id: %s: %s", chat_id, e)
            finally:
                queue.task_done()
                logger.debug("Task done for chat_id: %s", chat_id)
