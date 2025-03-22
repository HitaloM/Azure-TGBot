# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from aiogram.types import User
from azure.ai.inference.models import SystemMessage
from babel import Locale, UnknownLocaleError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_base_message() -> str:
    """
    Retrieves the base system message from a file.

    This function reads the content of the file located at "data/system.txt" and
    returns it as a string. If an error occurs during file reading, it logs the
    error and returns a default fallback message: "You are an AI assistant."

    Returns:
        str: The base system message if successfully read from the file, or a
        default fallback message in case of an error.
    """
    try:
        return Path("data/system.txt").read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Error reading system file: %s", exc)
        return "You are an AI assistant."


def get_system_message(user: User) -> SystemMessage:
    """
    Generates a system message containing user-specific and session-related information.

    Args:
        user (User): The user object containing details such as full name and language code.

    Returns:
        SystemMessage: A system message object with the base message and session information.

    The session information includes:
        - Current UTC date and time.
        - User's full name.
        - User's language and language code.
        - User's region (if available).

    If the user's language code is invalid or cannot be parsed, "Unknown" will be used for
    the language and region fields.
    """
    base_message = get_base_message()
    current_utc = datetime.now(UTC).strftime("%d-%m-%Y %H:%M:%S")
    language_code = user.language_code or "Unknown"

    try:
        user_locale = Locale.parse(language_code.replace("-", "_"))
        user_lang = user_locale.get_display_name() or "Unknown"
        user_region = user_locale.territory_name if user_locale.territory else "Unknown"
    except UnknownLocaleError:
        user_lang = "Unknown"
        user_region = "Unknown"

    session_info = (
        f"UTC Date and Time: {current_utc}\n"
        f"User Full Name: {user.full_name}\n"
        f"User Language: {user_lang} ({language_code})\n"
        f"User Region: {user_region}"
    )

    return SystemMessage(content=f"{base_message}\n_session:\n{session_info}")
