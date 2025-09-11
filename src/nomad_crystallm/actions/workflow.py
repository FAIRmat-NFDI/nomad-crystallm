from datetime import timedelta

from temporalio import workflow


with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.actions.activities import (
        construct_model_input,
        get_model,
        run_inference,
        write_results,
    )
    from nomad_crystallm.actions.shared import (
        InferenceModelInput,
        InferenceResults,
        InferenceResultsInput,
        InferenceUserInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> InferenceResults:
        constructed_model_input = await workflow.execute_activity(
            construct_model_input,
            data.prompt_generation_inputs,
            start_to_close_timeout=timedelta(seconds=60),
        )
        model_data = InferenceModelInput(
            prompts=constructed_prompts,
            inference_settings=data.inference_settings,
        )
        await workflow.execute_activity(
            get_model,
            model_data,
            start_to_close_timeout=timedelta(seconds=600),
        )
        generated_compositions_samples = await workflow.execute_activity(
            run_inference,
            model_data,
            start_to_close_timeout=timedelta(seconds=600),
        )
        for i, generated_samples in enumerate(generated_compositions_samples):
            await workflow.execute_activity(
                write_results,
                InferenceResultsInput(
                    user_id=data.user_id,
                    upload_id=data.upload_id,
                    action_instance_id=workflow.info().workflow_id,
                    composition=data.prompt_generation_inputs[i].input_composition,
                    prompt=model_data.prompts[i],
                    inference_settings=model_data.inference_settings,
                    generated_samples=generated_samples,
                    generate_cif=data.inference_settings.generate_cif,
                    relative_cif_dir=(
                        f'composition_{i + 1}_'
                        f'{data.prompt_generation_inputs[i].input_composition}'
                    ),
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )
        cif_paths = await workflow.execute_activity(
            write_results,
            InferenceResultsInput(
                user_id=data.user_id,
                upload_id=data.upload_id,
                generated_samples=generated_samples,
                generate_cif=data.generate_cif,
                model_data=model_data,
                cif_dir=workflow.info().workflow_id,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        return InferenceResults(
            cif_paths=cif_paths, generated_samples=generated_samples
        )
