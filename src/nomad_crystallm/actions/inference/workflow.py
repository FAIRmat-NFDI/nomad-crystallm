from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.actions.inference.activities import (
        get_model,
        get_prompt,
        run_inference,
        write_results,
    )
    from nomad_crystallm.actions.inference.models import (
        CrystallmUserInput,
        InferenceInput,
        InferenceOutput,
        WriteResultsInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> InferenceOutput:
        generated_samples = await workflow.execute_activity(
            run_inference,
            data,
            start_to_close_timeout=timedelta(hours=24),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return InferenceOutput(generated_samples=generated_samples)


@workflow.defn
class CrystallmWorkflow:
    @workflow.run
    async def run(self, data: CrystallmUserInput) -> None:
        retry_policy = RetryPolicy(maximum_attempts=3)
        await workflow.execute_activity(
            get_model,
            data.inference_settings.model,
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=retry_policy,
        )

        prompts = []  # list of all prompts along with as many duplicates as num_samples
        for prompt_construction_input in data.prompt_construction_inputs:
            prompt = await workflow.execute_activity(
                get_prompt,
                prompt_construction_input,
                start_to_close_timeout=timedelta(hours=1),
                retry_policy=retry_policy,
            )
            prompts.extend([prompt] * data.inference_settings.num_samples)

        num_batches = len(prompts) // data.inference_settings.batch_size
        if len(prompts) % data.inference_settings.batch_size != 0:
            num_batches += 1

        generated_samples_all_batches = []
        for batch_idx in range(num_batches):
            # create a child workflow for each batch
            range_start = batch_idx * data.inference_settings.batch_size
            range_end = min(
                (batch_idx + 1) * data.inference_settings.batch_size, len(prompts)
            )
            batch_prompts = prompts[range_start:range_end]
            inference_output = await workflow.execute_child_workflow(
                InferenceWorkflow.run,
                InferenceInput(
                    user_id=data.user_id,
                    upload_id=data.upload_id,
                    prompts=batch_prompts,
                    inference_settings=data.inference_settings,
                ),
                id=f'{workflow.info().workflow_id}_inference_batch_{batch_idx}',
                parent_close_policy=workflow.ParentClosePolicy.TERMINATE,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            generated_samples_all_batches.extend(inference_output.generated_samples)

        for idx, prompt_construction_input in enumerate(
            data.prompt_construction_inputs
        ):
            prompt = prompts[idx * data.inference_settings.num_samples]
            generated_samples = generated_samples_all_batches[
                idx * data.inference_settings.num_samples : (idx + 1)
                * data.inference_settings.num_samples
            ]
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
