from nomad.actions import TaskQueue
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class CrystaLLMInferenceEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-crystallm inference action.
    """

    def load(self):
        from nomad.actions import Action

        from nomad_crystallm.actions.inference.activities import (
            get_model,
            get_prompt,
            run_inference,
            write_results,
        )
        from nomad_crystallm.actions.inference.workflow import InferenceWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=InferenceWorkflow,
            activities=[get_model, get_prompt, run_inference, write_results],
        )


crystallm_inference = CrystaLLMInferenceEntryPoint(
    name='CrystaLLM Inference',
    task_queue=TaskQueue.CPU,
    description='Perform inference using the CrystaLLM model to generate crystal '
    'structures from chemical compositions.',
)
