from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad_crystallm.actions.bandgap_predictor.activities import (
        run_prediction_activity,
        setup_model,
    )
    from nomad_crystallm.actions.bandgap_predictor.shared import (
        BandGapPredictionInput,
        BandGapPredictionOutput,
    )


@workflow.defn
class BandGapPredictionWorkflow:
    @workflow.run
    async def run(
        self, input_data: BandGapPredictionInput
    ) -> list[BandGapPredictionOutput]:
        """
        The main workflow for the bandgap prediction.
        """
        checkpoint_url = 'https://raw.githubusercontent.com/vertaix/LLM-Prop/refs/heads/main/checkpoints/samples/classification/best_checkpoint_for_is_gap_direct.tar.gz'

        model_path = await workflow.execute_activity(
            setup_model,
            args=[
                checkpoint_url,
            ],
            start_to_close_timeout=timedelta(seconds=600),
        )
        predictions = await workflow.execute_activity(
            run_prediction_activity,
            args=[model_path, input_data],
            start_to_close_timeout=timedelta(seconds=600),
        )

        return predictions
