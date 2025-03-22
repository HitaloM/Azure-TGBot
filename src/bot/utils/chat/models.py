# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from enum import StrEnum


class AIModel(StrEnum):
    """
    AIModel is an enumeration that represents various AI model identifiers.

    Attributes:
        GPT_4O (str): Identifier for the GPT-4O model.
        GPT_4O_MINI (str): Identifier for the GPT-4O-Mini model.
        O1 (str): Identifier for the O1 model.
        O1_MINI (str): Identifier for the O1-Mini model.
        O3_MINI (str): Identifier for the O3-Mini model.
        DEEPSEEK_V3 (str): Identifier for the DeepSeek-V3 model.
        DEEPSEEK_R1 (str): Identifier for the DeepSeek-R1 model.
    """

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"

    # DeepSeek models
    DEEPSEEK_V3 = "DeepSeek-V3"
    DEEPSEEK_R1 = "DeepSeek-R1"

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return a list of all AI model identifiers.

        Returns:
            list[str]: A list containing the string values of all defined AI models.
        """
        return [model.value for model in cls]
