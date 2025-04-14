# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from aiogram.types import User
from azure.ai.inference.models import SystemMessage

from bot.utils.user_info import LocaleInfo, get_user_locale_info

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_base_message() -> str:
    """
    Retrieves the base system message from a file.

    Returns:
        str: The base system message if successfully read from the file, or a
        default fallback message in case of an error.
    """
    system_file_path = Path("data/system.txt")
    try:
        return system_file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        logger.error("System file not found: %s", exc)
        return "You are an AI assistant."
    except OSError as exc:
        logger.error("Error reading system file: %s", exc)
        return "You are an AI assistant."


def format_session_info(user: User, current_time: str, lang_info: LocaleInfo) -> str:
    """
    Formats session information into a structured string.

    Args:
        user (User): The user object.
        current_time (str): The current UTC time as a string.
        lang_info (LocaleInfo): The user's language and region information.

    Returns:
        str: A formatted string containing session information.
    """
    return (
        f"UTC Date and Time: {current_time}\n"
        f"User Full Name: {user.full_name}\n"
        f"User Language: {lang_info.language} ({user.language_code or 'Unknown'})\n"
        f"User Region: {lang_info.region}"
    )


def get_system_message(user: User) -> SystemMessage:
    """
    Generates a system message containing user-specific and session-related information.

    Args:
        user (User): The user object containing details such as full name and language code.

    Returns:
        SystemMessage: A system message object with the base message and session information.
    """
    if not user or not user.full_name:
        logger.warning("Invalid user provided to get_system_message")
        return SystemMessage(content=get_base_message())

    base_message = get_base_message()
    current_utc = datetime.now(UTC).strftime("%d-%m-%Y %H:%M:%S")
    language_code = user.language_code or "Unknown"
    lang_info = get_user_locale_info(language_code)
    session_info = format_session_info(user, current_utc, lang_info)

    return SystemMessage(content=f"{base_message}\n_session:\n{session_info}")
