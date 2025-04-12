# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>


from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Settings class for configuration that uses Pydantic BaseSettings.

    Attributes:
        bot_token (SecretStr): Telegram bot token.
        azure_api_key (SecretStr): Azure API key for accessing the inference service.
        azure_endpoint (AnyHttpUrl): URL endpoint for Azure Inference services.
        sudoers (list[int]): List of user IDs with sudo privileges.
    """

    bot_token: SecretStr
    azure_api_key: SecretStr
    azure_endpoint: AnyHttpUrl = AnyHttpUrl("https://models.inference.ai.azure.com")
    bing_api_key: SecretStr
    token_truncate_limit: int = 4000
    sudoers: list[int] = [918317361]

    @property
    def bot_id(self) -> str:
        """
        Retrieve the bot ID derived from the bot token.

        Returns:
            str: The bot ID, which is the first component of the bot token.
        """
        return self.bot_token.get_secret_value().split(":")[0]

    class Config:
        """
        Pydantic configuration for reading environment variables.
        """

        env_file = "data/config.env"
        env_file_encoding = "utf-8"


config = Settings()  # type: ignore[arg-type]
