# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import Settings

config = Settings()  # type: ignore

defaults = DefaultBotProperties(parse_mode=ParseMode.HTML, link_preview_is_disabled=True)
bot = Bot(token=config.bot_token.get_secret_value(), default=defaults)
