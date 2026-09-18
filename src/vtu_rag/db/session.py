from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vtu_rag.config import Settings
from vtu_rag.db.base import Base


class Database:
    """Owns the async engine and hands out sessions."""

    def __init__(self, settings: Settings):
        self.engine: AsyncEngine = create_async_engine(
            settings.postgres.url, pool_pre_ping=True, pool_size=5, max_overflow=10
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    # Columns added to existing tables after the first release. Derived data only,
    # so a plain ADD COLUMN is enough — no data migration, and re-indexing fills them.
    _ADDED_COLUMNS = (("figures", "label_text", "VARCHAR(512)"),)

    async def create_tables(self) -> None:
        # Import models so they register on Base.metadata
        from sqlalchemy import text

        import vtu_rag.models  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for table, column, column_type in self._ADDED_COLUMNS:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}")
                )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def ping(self) -> bool:
        from sqlalchemy import text

        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True

    async def close(self) -> None:
        await self.engine.dispose()
