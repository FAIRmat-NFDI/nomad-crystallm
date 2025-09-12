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
        write_prediction_results,
    )
    from nomad_crystallm.actions.crystallm_bandgap_predictor.shared import (
        CIFDescriptionInput,
        CrystaLLMBandGapPredictionOutput,
        WriteEntryInput,
    )
    from nomad_crystallm.actions.shared import (
        InferenceUserInput,
    )
    from nomad_crystallm.actions.workflow import InferenceWorkflow


@workflow.defn
class CrystaLLMBandGapPredictionWorkflow:
    @workflow.run
    async def run(self, data: InferenceUserInput) -> CrystaLLMBandGapPredictionOutput:
        """
        The main workflow for the CrystaLLM + Bandgap prediction.
        """
        workflow_id = workflow.info().workflow_id
        inference_workflow_id = f'crystallm-inference-workflow-{workflow_id}'
        inference_results = await workflow.execute_child_workflow(
            InferenceWorkflow.run, data, id=inference_workflow_id
        )

        description_output = await workflow.execute_activity(
            cif_to_description,
            CIFDescriptionInput(
                action_instance_id=inference_workflow_id,
                upload_id=data.upload_id,
                user_id=data.user_id,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )

        bandgap_input = BandGapPredictionInput(
            description_output=description_output,
            upload_id=data.upload_id,
            user_id=data.user_id,
        )

        bandgap_predictions = await workflow.execute_child_workflow(
            BandGapPredictionWorkflow.run,
            bandgap_input,
            id=f'bandgap-prediction-workflow-{workflow_id}',
        )

        print('time to execute results')
        await workflow.execute_activity(
            write_prediction_results,
            WriteEntryInput(
                user_id=data.user_id,
                upload_id=data.upload_id,
                prediction_outputs=bandgap_predictions,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        return CrystaLLMBandGapPredictionOutput(
            inference_results=inference_results,
            bandgap_predictions=bandgap_predictions,
        )
