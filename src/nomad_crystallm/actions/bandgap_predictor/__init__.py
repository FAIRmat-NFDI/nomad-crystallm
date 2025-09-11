from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class BandGapPredictorEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-crystallm bandgap prediction action.
    """

    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_crystallm.actions.bandgap_predictor.activities import (
            run_prediction_activity,
            setup_model,
        )
        from nomad_crystallm.actions.bandgap_predictor.workflow import (
            BandGapPredictionWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=BandGapPredictionWorkflow,
            activities=[setup_model, run_prediction_activity],
        )


bandgap_prediction = BandGapPredictorEntryPoint()
