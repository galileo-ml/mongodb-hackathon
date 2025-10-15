"""In-memory pub/sub for streaming conversation events over SSE."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Set

from ..core import ConversationEvent

logger = logging.getLogger("webrtc.conversation_stream")


class ConversationEventBus:
    """Async fan-out bus for conversation events."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue[ConversationEvent]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event: ConversationEvent) -> None:
        """Broadcast an event to all subscribers."""

        async with self._lock:
            targets = list(self._subscribers)
        if not targets:
            logger.debug(
                "No active subscribers; dropping %s event (person=%s conversation=%s)",
                event.event_type,
                event.person_id,
                event.conversation_id,
            )
            return

        for queue in targets:
            await queue.put(event)

    async def subscribe(self) -> asyncio.Queue[ConversationEvent]:
        """Register a new subscriber queue."""

        queue: asyncio.Queue[ConversationEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers.add(queue)
        logger.info("New conversation stream subscriber (total=%d)", len(self._subscribers))
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[ConversationEvent]) -> None:
        """Remove a subscriber and drain outstanding events."""

        async with self._lock:
            self._subscribers.discard(queue)
        while not queue.empty():
            queue.get_nowait()
        logger.info("Conversation stream subscriber removed (total=%d)", len(self._subscribers))

    async def stream(self) -> AsyncGenerator[ConversationEvent, None]:
        """Convenience generator yielding events for the caller."""

        queue = await self.subscribe()
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            await self.unsubscribe(queue)
