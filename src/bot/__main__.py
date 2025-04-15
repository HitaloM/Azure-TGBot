# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import logging

import uvloop
from aiogram import Dispatcher, Router
from aiogram.types import BotCommand, BotCommandScopeDefault
from aiogram.utils.chat_action import ChatActionMiddleware
from tortoise import Tortoise

from bot import bot

from .database.connection import init_db
from .filters.whitelist import WhiteListFilter
from .handlers.ask import router as ask_router
from .handlers.models import router as models_router
from .handlers.reset import router as reset_router
from .handlers.search import router as search_router
from .handlers.upgrade import router as upgrade_router
from .middlewares import setup_middlewares, shutdown_middlewares

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Main entry point for the bot application.

    This function initializes the database, sets up the dispatcher with filters,
    middlewares, and routers, configures bot commands, and starts polling for updates.

    It also ensures that database connections are closed upon termination.
    """
    await init_db()

    main_router = Router(name="main-router")
    dp = Dispatcher(name="root-dispatcher")

    main_router.message.filter(WhiteListFilter())

    setup_middlewares(main_router)

    main_router.message.middleware(ChatActionMiddleware())

    # Register routers in the proper order
    # Order matters - router are executed in the order they are included
    main_router.include_routers(
        reset_router, models_router, search_router, upgrade_router, ask_router
    )

    dp.include_router(main_router)

    bot_commands = [
        BotCommand(command="/ai", description="Interact with AI"),
        BotCommand(command="/search", description="Search the web with Bing"),
        BotCommand(command="/reset", description="Reset the conversation"),
    ]

    await bot.delete_my_commands()
    await bot.set_my_commands(bot_commands, scope=BotCommandScopeDefault(), request_timeout=30)

    logger.info("Starting the bot...")

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await shutdown_middlewares()
        await Tortoise.close_connections()


if __name__ == "__main__":
    try:
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(main())
    except KeyboardInterrupt:
        logger.warning("Forced stop... Bye!")
