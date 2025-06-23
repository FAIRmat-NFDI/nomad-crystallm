from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.workflows.activities import (
        construct_model_input,
        get_model,
        run_inference,
        write_results,
    )
    from nomad_crystallm.workflows.shared import (
        InferenceInput,
        InferenceModelInput,
        InferenceResultsInput,
    )


@workflow.defn(name='nomad_crystallm.workflows.InferenceWorkflow')
class InferenceWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> list[str]:
        raw_input = await workflow.execute_activity(
            construct_model_input,
            data,
            start_to_close_timeout=timedelta(seconds=60),
        )
        model_data = InferenceModelInput(
            raw_input=raw_input,
            model_path=data.model_path,
            model_url=data.model_url,
            num_samples=data.num_samples,
            max_new_tokens=data.max_new_tokens,
            temperature=data.temperature,
            top_k=data.top_k,
            seed=data.seed,
            dtype=data.dtype,
            compile=data.compile,
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
