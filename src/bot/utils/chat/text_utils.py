# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import re


class TextSplitter:
    """
    A utility class for splitting text into chunks of a specified maximum length.

    Attributes:
        max_length (int): The maximum length of each text chunk.
        chunks (list[str]): A list of text chunks that have been processed.
        current_chunk (list[str]): The current chunk being built.
        current_length (int): The current length of the text in the current chunk.

    Methods:
        flush_chunk() -> None:
            Finalizes the current chunk by appending it to the list of chunks and resets the
                current chunk.

        add_text(text: str) -> None:
            Adds a piece of text to the current chunk. Raises a ValueError if the text exceeds
                the maximum length.

        process_pre_token(token: str) -> None:
            Processes a token and adds it to the appropriate chunk. Handles cases where the
                token length exceeds the maximum length.

        process_line(line: str) -> None:
            Processes a line of text and splits it into chunks as needed. Handles cases where
                the line length exceeds the maximum length.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length
        self.chunks: list[str] = []
        self.current_chunk: list[str] = []
        self.current_length = 0

    def flush_chunk(self) -> None:
        """
        Finalizes the current chunk by appending it to the list of chunks.

        This method checks if there is any content in the current chunk. If so, it
        concatenates the content into a single string, appends it to the `chunks` list,
        and then clears the current chunk and resets its length tracker.
        """
        if self.current_chunk:
            self.chunks.append("".join(self.current_chunk))
            self.current_chunk.clear()
            self.current_length = 0

    def add_text(self, text: str) -> None:
        """
        Adds a text string to the current chunk if it does not exceed the maximum allowed length.

        If the text exceeds the `max_length` attribute, a `ValueError` is raised. If adding the
        text to the current chunk would exceed the `max_length`, the current chunk is flushed
        before appending the new text.

        Args:
            text (str): The text string to be added.

        Raises:
            ValueError: If the length of the text exceeds the `max_length` attribute.
        """
        if len(text) > self.max_length:
            msg = "Text exceeds max_length"
            raise ValueError(msg)

        if self.current_length + len(text) > self.max_length:
            self.flush_chunk()

        self.current_chunk.append(text)
        self.current_length += len(text)

    def process_pre_token(self, token: str) -> None:
        """
        Processes a token and determines how it should be handled based on its length
        relative to the maximum allowed length.

        If the token exceeds the maximum length, it is directly added to the chunks
        after flushing the current chunk. If adding the token to the current chunk
        would exactly match the maximum length, the token is added to the current
        chunk, and the chunk is then flushed. If adding the token would exceed the
        maximum length, the current chunk is flushed, and the token is processed
        recursively. Otherwise, the token is added to the current chunk.

        Args:
            token (str): The token to be processed.
        """
        if len(token) > self.max_length:
            self.flush_chunk()
            self.chunks.append(token)
        elif self.current_length + len(token) == self.max_length:
            self.current_chunk.append(token)
            self.flush_chunk()
        elif self.current_length + len(token) > self.max_length:
            self.flush_chunk()
            self.add_text(token)
        else:
            self.add_text(token)

    def process_line(self, line: str):
        """
        Processes a line of text and splits it into chunks based on a maximum length.

        If the line exceeds the maximum length, it is divided into smaller pieces
        of the specified maximum length. These pieces are either added directly to
        the chunks list or processed further. If the current accumulated length
        combined with the new line exceeds the maximum length, the current chunk
        is flushed before adding the new text.

        Args:
            line (str): The line of text to be processed.
        """
        if len(line) > self.max_length:
            if self.current_length:
                self.flush_chunk()
            idx = 0
            while idx < len(line):
                piece = line[idx : idx + self.max_length]
                if len(piece) == self.max_length:
                    self.chunks.append(piece)
                else:
                    self.add_text(piece)
                idx += self.max_length
        elif self.current_length + len(line) == self.max_length:
            self.add_text(line)
            self.flush_chunk()
        elif self.current_length + len(line) > self.max_length:
            self.flush_chunk()
            self.add_text(line)
        else:
            self.add_text(line)


def split_text_with_formatting(text: str, max_length: int = 4096) -> list[str]:
    """
    Splits a given text into smaller chunks while preserving formatting, such as preformatted
    text blocks.

    This function ensures that the text is split into chunks that do not exceed the specified
    maximum length. It handles preformatted text blocks (enclosed in `<pre>` tags) separately
    to preserve their structure, and splits other text into lines while maintaining line breaks.

    Args:
        text (str): The input text to be split.
        max_length (int, optional): The maximum length of each chunk. Defaults to 4096.

    Returns:
        list[str]: A list of text chunks, each adhering to the maximum length constraint.
    """
    splitter = TextSplitter(max_length)
    tokens = re.split(r"(<pre>.*?</pre>)", text, flags=re.DOTALL)
    for token in tokens:
        if token.startswith("<pre>") and token.endswith("</pre>"):
            splitter.process_pre_token(token)
        else:
            for line in token.splitlines(keepends=True):
                splitter.process_line(line)
    splitter.flush_chunk()
    return splitter.chunks
