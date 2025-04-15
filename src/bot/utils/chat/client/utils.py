# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import re


def extract_retry_seconds_from_error(error_message: str) -> int:
    """
    Extract retry wait time in seconds from an API error message.

    Args:
        error_message: Error message from the API

    Returns:
        Number of seconds to wait before retrying, or 3600 (1 hour) as default fallback
    """
    pattern = r"Please wait (\d+) seconds before retrying"
    match = re.search(pattern, error_message)

    if match:
        return int(match.group(1))

    # Return 1 hour as a conservative default if no specific time found
    return 3600
