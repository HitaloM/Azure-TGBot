# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from tortoise import Tortoise


async def init_db() -> None:
    """
    Initialize the database connection and generate schemas.

    This function sets up the database connection using Tortoise ORM with the specified
    SQLite database URL and model modules. It also generates the database schemas
    based on the defined models.

    Returns:
        None
    """
    await Tortoise.init(
        db_url="sqlite://data/db.sqlite3",
        modules={"models": ["bot.database.models"]},
    )
    await Tortoise.generate_schemas()
