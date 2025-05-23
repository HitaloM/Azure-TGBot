# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from enum import StrEnum


class AIModel(StrEnum):
    """
    AIModel is an enumeration that represents various AI model identifiers.

    Attributes:
        GPT_4_1 (str): Identifier for the GPT-4.1 model.
        GPT_4_1_MINI (str): Identifier for the GPT-4.1-Mini model.
        GPT_4_1_NANO (str): Identifier for the GPT-4.1-Nano model.
        O3 (str): Identifier for the O3 model.
        O4_MINI (str): Identifier for the O4-Mini model.
        DEEPSEEK_V3 (str): Identifier for the DeepSeek-V3-0324 model.
        DEEPSEEK_R1 (str): Identifier for the DeepSeek-R1 model.
        MAI_DS_R1 (str): Identifier for the MAI-DS-R1 model.
        GROK_3 (str): Identifier for the Grok-3 model.
    """

    # OpenAI models
    GPT_4_1 = "openai/gpt-4.1"
    GPT_4_1_MINI = "openai/gpt-4.1-mini"
    GPT_4_1_NANO = "openai/gpt-4.1-nano"
    O3 = "openai/o3"
    O4_MINI = "openai/o4-mini"

    # DeepSeek models
    DEEPSEEK_V3 = "deepseek/DeepSeek-V3-0324"
    DEEPSEEK_R1 = "deepseek/DeepSeek-R1"

    # Microsoft models
    MAI_DS_R1 = "microsoft/MAI-DS-R1"

    # xAI models
    GROK_3 = "xai/grok-3"
    GROK_3_MINI = "xai/grok-3-mini"

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return a list of all AI model identifiers.

        Returns:
            list[str]: A list containing the string values of all defined AI models.
        """
        return [model.value for model in cls]
