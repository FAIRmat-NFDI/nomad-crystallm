from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class CrystaLLMBandGapPredictionEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-crystallm bandgap prediction action.
    """

    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_crystallm.actions.crystallm_bandgap_predictor.activities import (
            cif_to_description,
        )
        from nomad_crystallm.actions.crystallm_bandgap_predictor.workflow import (
            CrystaLLMBandGapPredictionWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=CrystaLLMBandGapPredictionWorkflow,
            activities=[cif_to_description],
        )

crystallm_bandgap_prediction = CrystaLLMBandGapPredictionEntryPoint()
