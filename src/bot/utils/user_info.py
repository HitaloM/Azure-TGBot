# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from typing import NamedTuple

from babel import Locale, UnknownLocaleError


class LocaleInfo(NamedTuple):
    language: str
    region: str


def get_user_locale_info(language_code: str) -> LocaleInfo:
    """
    Extracts locale information from the user's language code.

    Args:
        language_code (str): The language code of the user.

    Returns:
        LocaleInfo: An object containing the user's language and region.
    """
    try:
        user_locale = Locale.parse(language_code.replace("-", "_"))
        user_lang = user_locale.get_display_name() or "Unknown"
        user_region = user_locale.territory_name if user_locale.territory else "Unknown"
        return LocaleInfo(language=user_lang, region=user_region)
    except UnknownLocaleError:
        return LocaleInfo(language="Unknown", region="Unknown")
