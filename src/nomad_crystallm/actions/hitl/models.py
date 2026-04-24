from typing import Literal

from pydantic import BaseModel, Field

from nomad.actions.assets.models import ActionAssetRef


class BaseWorkflowInput(BaseModel):
    """Base input model for workflows"""

    user_id: str = Field(
        ..., description='Unique identifier for the user who initiated the workflow.'
    )


class UserInputExampleWorkflowInput(BaseWorkflowInput):
    """Input model for the user input example workflow"""

    lower_bound: int = Field(
        ..., description='The lower bound for the random number generation.'
    )
    upper_bound: int = Field(
        ..., description='The upper bound for the random number generation.'
    )


class UserInputData(BaseModel):
    """Payload model for what the user submits when requested."""

    decision: Literal['approve', 'disapprove'] = Field(
        ..., description="The user's decision (e.g. 'approve', 'disapprove')"
    )
    notes: str = Field(default='', description='Optional notes provided by the user.')
    image_path: ActionAssetRef = Field(
        ...,
        description='Image file reference uploaded with the signal submission.',
        json_schema_extra={
            'x-nomad-widget': 'image-upload',
            'accept': ['image/*'],
        },
    )
    audio_submission: ActionAssetRef = Field(
        ...,
        description='Audio file reference uploaded with the signal submission.',
        json_schema_extra={
            'x-nomad-widget': 'audio-upload',
            'accept': ['audio/*'],
        },
    )
