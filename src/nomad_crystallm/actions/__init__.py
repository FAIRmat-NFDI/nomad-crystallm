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
    num_prompts_per_action: int = Field(
        default=1, description='Number of prompts to process in one action instance.'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_crystallm.actions.activities import (
            get_model,
            get_prompt,
            limit_prompts
            run_inference,
            write_results,
        )
        from nomad_crystallm.actions.workflow import InferenceWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=InferenceWorkflow,
            activities=[
                get_model,
                get_prompt,
                limit_prompts,
                run_inference,
                write_results,
            ],
        )


crystallm_inference = CrystaLLMInferenceEntryPoint(
    name='CrystaLLM Inference',
    plugin_package='FAIRmat-NFDI/nomad-crystallm',
    description='Perform inference using the CrystaLLM model to generate crystal '
    'structures from chemical compositions.',
)
