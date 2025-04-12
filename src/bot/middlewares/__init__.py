# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging

from aiogram import Router

from src.bot.middlewares.queue import QueueMiddleware
from src.bot.middlewares.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)

# Keep track of middleware instances to allow cleanup on shutdown
_middleware_instances = []


def setup_middlewares(router: Router) -> None:
    """
    Configures and registers middlewares for the provided router.

    This function sets up two middlewares:
    1. QueueMiddleware: Manages a queue for processing messages with specific
       constraints such as maximum queue size, processing timeout, cleanup
       interval, and idle timeout.
    2. RateLimitMiddleware: Enforces rate limits for users and groups, defining
       the maximum number of requests allowed within specified time intervals.

    The middlewares are registered in the appropriate order, as the execution
    order depends on the registration sequence. Middleware instances are also
    stored for later shutdown.

    Args:
        router (Router): The router instance to which the middlewares will be
        attached.

    Raises:
        None
    """
    queue_middleware = QueueMiddleware(
        max_queue_size=50,  # Maximum of 50 messages in the queue
        process_timeout=60.0,  # Timeout of 60 seconds per message
        cleanup_interval=300,  # Inactivity check every 5 minutes
        idle_timeout=3600,  # Remove inactive queue after 1 hour
    )

    rate_limit_middleware = RateLimitMiddleware(
        user_limit=5,  # 5 requests per user
        user_interval=30,  # 30-second interval per user
        group_limit=10,  # 10 requests per group
        group_interval=60,  # 60-second interval per group
    )

    # Register middlewares in the proper order
    # Order matters - middlewares are executed in the order they are registered

    router.message.outer_middleware(rate_limit_middleware)
    router.callback_query.outer_middleware(rate_limit_middleware)
    logger.debug("Rate limit middleware registered")

    router.message.outer_middleware(queue_middleware)
    logger.debug("Queue middleware registered")

    # Store middleware instances for later shutdown
    _middleware_instances.extend([rate_limit_middleware, queue_middleware])
    logger.info("All middlewares registered successfully")


async def shutdown_middlewares():
    """
    Asynchronously shuts down all middleware instances.

    This function iterates through all middleware instances stored in the
    `_middleware_instances` list and attempts to call their `_shutdown` method
    if it exists. It logs the shutdown process, including any errors encountered
    while shutting down individual middleware instances.

    Raises:
        Any exceptions raised during the shutdown of a middleware are caught and logged,
        but the function continues shutting down the remaining middlewares.
    """
    logger.info("Shutting down middlewares...")

    for middleware in _middleware_instances:
        if hasattr(middleware, "_shutdown"):
            try:
                await middleware._shutdown()
                logger.debug("Middleware %s shut down correctly", middleware.__class__.__name__)
            except Exception as e:
                logger.error(
                    "Error shutting down middleware %s: %s",
                    middleware.__class__.__name__,
                    e,
                )

    logger.info("All middlewares have been shut down")
