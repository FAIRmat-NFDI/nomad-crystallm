from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class CrystaLLMInferenceEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-crystallm inference action.
    """

    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_crystallm.workflows.activities import (
            construct_model_input,
            get_model,
            run_inference,
            write_results,
        )
        from nomad_crystallm.workflows.workflow import InferenceWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=InferenceWorkflow,
            activities=[get_model, construct_model_input, run_inference, write_results],
        )


crystallm_inference = CrystaLLMInferenceEntryPoint()
