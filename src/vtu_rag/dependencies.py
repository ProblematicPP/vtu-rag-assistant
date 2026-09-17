from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from vtu_rag.agent import AgentService
from vtu_rag.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_agent(request: Request) -> AgentService:
    return request.app.state.agent


async def get_session(
    container: Annotated[Container, Depends(get_container)],
) -> AsyncIterator[AsyncSession]:
    async with container.db.session() as session:
        yield session


ContainerDep = Annotated[Container, Depends(get_container)]
AgentDep = Annotated[AgentService, Depends(get_agent)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
