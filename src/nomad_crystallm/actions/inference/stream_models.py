# TODO: Drop this file once the new NOMAD release including action streams is out.
# These models are replicated from nomad.actions.models to avoid dependencies
# on new nomad versions.

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

try:
    from nomad.actions.models import (
        ActionStreamEvent,
        ActionStreamEventSeverity,
        ActionStreamEventType,
    )
    from nomad.actions.streams import ACTION_STREAM_TOPIC, action_event_publisher
except ImportError:
    ACTION_STREAM_TOPIC = 'action'

    class ActionStreamEventType(str, Enum):
        """Generic event categories that action UIs can handle consistently."""

        STATE = 'state'
        MESSAGE = 'message'
        OUTPUT_DELTA = 'output_delta'

    class ActionStreamEventSeverity(str, Enum):
        """User-facing severity for action stream events."""

        INFO = 'info'
        SUCCESS = 'success'
        WARNING = 'warning'
        ERROR = 'error'

    class ActionStreamEvent(BaseModel):
        """Structured event that plugin workflows and activities can stream."""

        type: ActionStreamEventType = Field(
            ..., description='Generic event category for frontend rendering.'
        )
        name: str | None = Field(
            default=None,
            description='Optional plugin-specific event name, e.g. search_started.',
        )
        operation_id: str | None = Field(
            default=None,
            description='Optional identity for merging updates from one operation.',
        )
        message: str | None = Field(
            default=None, description='Short user-facing event message.'
        )
        progress: float | None = Field(
            default=None,
            ge=0,
            le=100,
            description='Optional progress percentage from 0 to 100.',
        )
        data: dict[str, Any] = Field(
            default_factory=dict,
            description='Optional structured data for event-specific UI rendering.',
        )
        severity: ActionStreamEventSeverity = Field(
            default=ActionStreamEventSeverity.INFO,
            description='User-facing event severity.',
        )
        terminal: bool = Field(
            default=False, description='True when this event marks the stream terminal.'
        )
        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description='Event creation timestamp.',
        )

    @asynccontextmanager
    async def action_event_publisher(
        workflow_id: str | None = None,
        *,
        batch_interval: timedelta = timedelta(milliseconds=200),
        max_batch_size: int = 100,
        max_retry_duration: timedelta = timedelta(minutes=10),
    ):
        from temporalio.contrib.workflow_streams import WorkflowStreamClient

        if workflow_id is None:
            stream_client = WorkflowStreamClient.from_within_activity(
                batch_interval=batch_interval,
                max_batch_size=max_batch_size,
                max_retry_duration=max_retry_duration,
            )
        else:
            from temporalio import activity

            stream_client = WorkflowStreamClient.create(
                activity.client(),
                workflow_id=workflow_id,
                batch_interval=batch_interval,
                max_batch_size=max_batch_size,
                max_retry_duration=max_retry_duration,
            )

        async with stream_client:
            yield stream_client.topic(ACTION_STREAM_TOPIC, type=ActionStreamEvent)
