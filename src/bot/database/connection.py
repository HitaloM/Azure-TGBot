# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = "sqlite+aiosqlite:///data/db.sqlite3"

SQLITE_URL_PARAMS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "foreign_keys": "ON",
    "cache_size": "-5000",
}

if DATABASE_URL.startswith("sqlite"):
    query_params = "&".join(f"{k}={v}" for k, v in SQLITE_URL_PARAMS.items())
    if "?" not in DATABASE_URL:
        DATABASE_URL = f"{DATABASE_URL}?{query_params}"
    else:
        DATABASE_URL = f"{DATABASE_URL}&{query_params}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=3600,
)

async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models"""

    pass


async def optimize_sqlite(engine: AsyncEngine) -> None:
    """
    Executa comandos de otimização no banco de dados SQLite.

    Isso inclui análise de estatísticas e VACUUM para manter o banco de dados otimizado.

    Args:
        engine: A engine do SQLAlchemy para executar os comandos.
    """
    async with engine.begin() as conn:
        # Analisa estatísticas do banco para otimizar consultas
        await conn.execute(text("ANALYZE"))

        # Executa VACUUM para reorganizar o banco e recuperar espaço
        # O modo incremental é mais eficiente para operações frequentes
        await conn.execute(text("VACUUM"))


async def init_db() -> None:
    """
    Initialize the database connection and apply migrations.

    This function sets up the database connection using SQLAlchemy async with the specified
    SQLite database URL. It also applies any pending migrations using Alembic.

    Returns:
        None
    """
    project_dir = Path(__file__).parent.parent.parent.parent
    process = await asyncio.create_subprocess_exec(
        "alembic",
        "upgrade",
        "head",
        cwd=str(project_dir),
    )
    await process.communicate()

    # Otimiza o banco de dados após as migrações
    await optimize_sqlite(engine)


# Função para agendar VACUUM periódico
async def schedule_vacuum_optimization(interval_hours: int = 24) -> None:
    """
    Agenda a execução periódica de VACUUM e outras otimizações.

    Args:
        interval_hours: Intervalo em horas entre cada otimização.
    """
    while True:
        await asyncio.sleep(interval_hours * 3600)  # Converte horas para segundos
        await optimize_sqlite(engine)
