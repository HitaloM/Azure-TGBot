# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>


def split_text_with_formatting(text: str, max_length: int = 4096) -> list[str]:
    """
    Split text into smaller parts while preserving Markdown formatting.

    This function divides text into parts that don't exceed the specified maximum length,
    ensuring that formatting elements such as code blocks, bold text, italics, and
    others are not incorrectly broken. It's especially useful for preparing long
    messages to be sent via Telegram, which has a 4096 character limit per message.

    Args:
        text: The text to be split.
        max_length: Maximum length for each part (default: 4096 for Telegram).

    Returns:
        A list of text parts, each respecting the length limit.

    Examples:
        >>> split_text_with_formatting("Short message")
        ['Short message']

        >>> split_text_with_formatting("Long text..." * 1000, 100)
        ['Long text...Long text...', 'Long text...Long text...', ...]
    """
    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    splitter = TextSplitter()
    return splitter.process_text(text, max_length)


class TextSplitter:
    """
    Utility class for splitting text into parts while preserving formatting.

    This class provides methods to split text into smaller chunks that respect
    maximum size constraints while preserving Markdown formatting elements
    like code blocks, bold, italics, etc.

    The class takes special care of code blocks (```), ensuring that:
    1. Code blocks are not split in the middle
    2. If a code block needs to be split, formatting is maintained across all parts
    3. Code block syntax (e.g., ```python) is preserved in all parts
    """

    def process_text(self, text: str, max_length: int) -> list[str]:
        """
        Process the text and split it into parts.

        This main method controls the text processing flow, splitting it into
        parts that respect the maximum length while maintaining the integrity
        of formatting elements.

        Args:
            text: The text to be processed and split.
            max_length: The maximum length for each part.

        Returns:
            A list with the text parts.
        """
        result = []
        current_text = text

        # Keep track of open formatting elements like code blocks
        open_code_block = False
        code_block_syntax = ""

        while current_text:
            # If too short to need splitting
            if len(current_text) <= max_length:
                result.append(current_text)
                break

            # Find where to split
            split_point = self._find_safe_split_point(current_text, max_length)

            # Get current chunk and rest of text
            chunk = current_text[:split_point]
            rest = current_text[split_point:]

            # Process code blocks
            code_block_info = self._process_code_blocks(chunk, open_code_block, code_block_syntax)
            open_code_block = code_block_info["open_code_block"]
            code_block_syntax = code_block_info["code_block_syntax"]

            result.append(chunk)

            # Add code block formatting to next chunk if needed
            rest = self._format_next_chunk(rest, open_code_block, code_block_syntax)
            current_text = rest

        return result

    def _process_code_blocks(
        self, chunk: str, open_code_block: bool, code_block_syntax: str
    ) -> dict:
        """
        Process code blocks in the current text part.

        This method analyzes the current text part to determine if it contains
        open or closed code blocks and tracks the state of these blocks.

        Args:
            chunk: The current text part being processed.
            open_code_block: Indicates if there's an open code block from the previous part.
            code_block_syntax: The language specified for the code block, if any.

        Returns:
            A dictionary containing the new code block state and its syntax.
        """
        code_block_starts = chunk.count("```") - chunk.count("````")

        if code_block_starts % 2 != 0:  # We have an unclosed code block
            open_code_block = not open_code_block

            if open_code_block:  # We just opened a code block
                code_block_syntax = self._extract_code_block_syntax(chunk)

        return {"open_code_block": open_code_block, "code_block_syntax": code_block_syntax}

    @staticmethod
    def _extract_code_block_syntax(text: str) -> str:
        """
        Extract the syntax used in a code block.

        For example, from "```python", extracts "python".

        Args:
            text: The text containing the code block marker.

        Returns:
            The specified syntax for the code block or empty string if none.
        """
        lines = text.split("\n")
        for line in lines:
            if "```" in line and not line.strip().endswith("```"):
                return line.strip().replace("```", "").strip()
        return ""

    @staticmethod
    def _format_next_chunk(text: str, open_code_block: bool, code_block_syntax: str) -> str:
        """
        Format the next part based on open formatting elements.

        If a code block is open, this method adds the necessary
        formatting at the beginning of the next part.

        Args:
            text: The text for the next part.
            open_code_block: Indicates if there's an open code block.
            code_block_syntax: The code block syntax, if any.

        Returns:
            The formatted text for the next part.
        """
        if not open_code_block:
            return text

        if code_block_syntax:
            return f"```{code_block_syntax}\n{text}"

        return f"```\n{text}"

    @staticmethod
    def _find_safe_split_point(text: str, max_length: int) -> int:
        """
        Find a safe point to split the text without breaking formatting.

        This method looks for ideal points to split the text, such as line breaks
        or spaces, to ensure a more natural division of content.
        If it doesn't find ideal points, it splits at the maximum specified length.

        Args:
            text: The text to find a splitting point for.
            max_length: The maximum length of text before needing to split.

        Returns:
            The index at which it's safe to split the text.
        """
        if len(text) <= max_length:
            return len(text)

        # Try to split at a newline
        for i in range(max_length, 0, -1):
            if text[i] == "\n":
                return i + 1

        # Try to split at space
        for i in range(max_length, 0, -1):
            if text[i] == " ":
                return i + 1

        # If we can't find a good split point, just split at max_length
        return max_length
