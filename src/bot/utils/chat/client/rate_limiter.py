# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from datetime import UTC, datetime, timedelta

from bot.utils.chat.models import AIModel

logger = logging.getLogger(__name__)


class RateLimitTracker:
    """
    RateLimitTracker is a utility class for managing and tracking rate limits for different
    AI models.

    This class allows you to:
    - Check if a specific model is currently rate limited.
    - Set a rate limit for a model for a specified duration.
    - Retrieve the remaining wait time before a model is no longer rate limited.

    Attributes:
        rate_limited_models (dict[AIModel, datetime]):
            A mapping of AI models to the datetime until which they are rate limited.
    """

    def __init__(self):
        self.rate_limited_models: dict[AIModel, datetime] = {}

    def is_rate_limited(self, model: AIModel) -> bool:
        """
        Checks if the specified AI model is currently rate limited.

        Args:
            model (AIModel): The AI model to check for rate limiting.

        Returns:
            bool: True if the model is rate limited (i.e., the current time is before the rate
            limit expiration), False otherwise.
        """
        if model not in self.rate_limited_models:
            return False

        return datetime.now(tz=UTC) < self.rate_limited_models[model]

    def set_rate_limited(self, model: AIModel, seconds: int):
        """
        Sets a rate limit for the specified AI model by recording the time until which the
        model is rate limited.

        Args:
            model (AIModel): The AI model to be rate limited.
            seconds (int): The number of seconds for which the model should be rate limited.
        """
        self.rate_limited_models[model] = datetime.now(tz=UTC) + timedelta(seconds=seconds)
        logger.info(
            "Model %s will be rate limited until %s", model.value, self.rate_limited_models[model]
        )

    def get_wait_time(self, model: AIModel) -> int:
        """
        Calculates the remaining wait time in seconds before requests to the specified
        AI model are allowed, based on rate limiting.

        Args:
            model (AIModel): The AI model for which to check the wait time.

        Returns:
            int: The number of seconds to wait before making a request to the model.
            Returns 0 if not rate limited.
        """
        if not self.is_rate_limited(model):
            return 0

        remaining = (self.rate_limited_models[model] - datetime.now(tz=UTC)).total_seconds()
        return max(0, int(remaining))


rate_limit_tracker = RateLimitTracker()
