from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.actions.bandgap_predictor.shared import (
        BandGapPredictionInput,
    )
    from nomad_crystallm.actions.bandgap_predictor.workflow import (
        BandGapPredictionWorkflow,
    )
    from nomad_crystallm.actions.crystallm_bandgap_predictor.activities import (
        cif_to_description,
    )
    from nomad_crystallm.actions.crystallm_bandgap_predictor.shared import (
        CIFDescriptionInput,
        CrystaLLMBandGapPredictionOutput,
    )
    from nomad_crystallm.actions.shared import (
        InferenceInput,
    )
    from nomad_crystallm.actions.workflow import InferenceWorkflow


@workflow.defn
class CrystaLLMBandGapPredictionWorkflow:
    @workflow.run
    async def run(self, data: InferenceInput) -> CrystaLLMBandGapPredictionOutput:
        """
        The main workflow for the CrystaLLM + Bandgap prediction.
        """
        workflow_id = workflow.info().workflow_type
        inference_results = await workflow.execute_child_workflow(
            InferenceWorkflow.run,
            data,
            id=f'crystallm-inference-workflow-{workflow_id}',
        )

        descriptions = []
        for cif_path in inference_results.cif_paths:
            description = await workflow.execute_activity(
                cif_to_description,
                CIFDescriptionInput(
                    cif_path=cif_path, upload_id=data.upload_id, user_id=data.user_id
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )
            descriptions.append(description)

        bandgap_input = BandGapPredictionInput(
            descriptions=descriptions,
            upload_id=data.upload_id,
            user_id=data.user_id,
        )

        bandgap_predictions = await workflow.execute_child_workflow(
            BandGapPredictionWorkflow.run,
            bandgap_input,
            id=f'bandgap-prediction-workflow-{workflow_id}',
        )

        return CrystaLLMBandGapPredictionOutput(
            generated_samples=inference_results.generated_samples,
            bandgap_predictions=bandgap_predictions,
        )
