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
        InferenceResultsInput,
        InferenceUserInput,
    )


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceUserInput) -> list[str]:
        constructed_model_input = await workflow.execute_activity(
            construct_model_input,
            data,
            start_to_close_timeout=timedelta(seconds=60),
        )
        model_data = InferenceModelInput(
            prompts=constructed_model_input,
            inference_settings=data.inference_settings,
        )
        await workflow.execute_activity(
            get_model,
            model_data,
            start_to_close_timeout=timedelta(seconds=600),
        )
        generated_samples = await workflow.execute_activity(
            run_inference,
            model_data,
            start_to_close_timeout=timedelta(seconds=600),
        )
        await workflow.execute_activity(
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
        return generated_samples
