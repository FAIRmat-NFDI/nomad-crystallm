from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.actions.activities import (
        construct_prompts,
        get_model,
        run_inference,
        write_results,
    )
    from nomad_crystallm.actions.shared import (
        ConstructPromptInput,
        InferenceInput,
        InferenceUserInput,
        WriteResultsInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceUserInput) -> None:
        constructed_prompts = await workflow.execute_activity(
            construct_prompts,
            ConstructPromptInput(
                prompter=data.prompter,
                upload_id=data.upload_id,
                user_id=data.user_id,
            ),
            start_to_close_timeout=timedelta(hours=1),
        )
        await workflow.execute_activity(
            get_model,
            data.inference_settings.model_name,
            start_to_close_timeout=timedelta(hours=1),
        )
        for idx, constructed_prompt in enumerate(constructed_prompts):
            inference_output = await workflow.execute_activity(
                run_inference,
                InferenceInput(
                    constructed_prompt=constructed_prompt,
                    inference_settings=data.inference_settings,
                ),
                start_to_close_timeout=timedelta(hours=1),
            )
            await workflow.execute_activity(
                write_results,
                WriteResultsInput(
                    user_id=data.user_id,
                    upload_id=data.upload_id,
                    action_instance_id=workflow.info().workflow_id,
                    relative_cif_dir=(
                        f'composition_{idx + 1}_{constructed_prompt.composition}'
                    ),
                    constructed_prompt=constructed_prompt,
                    inference_settings=data.inference_settings,
                    inference_output=inference_output,
                ),
                start_to_close_timeout=timedelta(hours=1),
            )
