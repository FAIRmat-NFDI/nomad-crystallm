from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.actions.activities import (
        get_model,
        get_prompt,
        run_inference,
        write_results,
    )
    from nomad_crystallm.actions.shared import (
        InferenceInput,
        InferenceUserInput,
        WriteResultsInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceUserInput) -> None:
        retry_policy = RetryPolicy(maximum_attempts=3)
        await workflow.execute_activity(
            get_model,
            data.inference_settings.model,
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=retry_policy,
        )
        for idx, prompt_construction_input in enumerate(
            data.prompt_construction_inputs
        ):
            prompt = await workflow.execute_activity(
                get_prompt,
                prompt_construction_input,
                start_to_close_timeout=timedelta(hours=1),
                retry_policy=retry_policy,
            )
            generated_samples = await workflow.execute_activity(
                run_inference,
                InferenceInput(
                    prompt=prompt,
                    inference_settings=data.inference_settings,
                ),
                start_to_close_timeout=timedelta(hours=1),
                retry_policy=retry_policy,
            )
            await workflow.execute_activity(
                write_results,
                WriteResultsInput(
                    user_id=data.user_id,
                    upload_id=data.upload_id,
                    action_instance_id=workflow.info().workflow_id,
                    relative_cif_dir=(
                        f'composition_{idx + 1}_{prompt_construction_input.composition}'
                    ),
                    composition=prompt_construction_input.composition,
                    prompt=prompt,
                    inference_settings=data.inference_settings,
                    generated_samples=generated_samples,
                ),
                start_to_close_timeout=timedelta(hours=1),
                retry_policy=retry_policy,
            )
