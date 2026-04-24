from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.actions.manager import (
        request_signal_input,
    )

    from nomad_crystallm.actions.hitl.activities import (
        collect_signal_asset_metadata_activity,
        generate_random_number_activity,
    )
    from nomad_crystallm.actions.hitl.models import (
        UserInputData,
        UserInputExampleWorkflowInput,
    )


@workflow.defn(name='nomad_example.actions.workflows.UserInputExampleWorkflow')
class UserInputExampleWorkflow:
    def __init__(self) -> None:
        self._user_input: UserInputData | None = None

    @workflow.signal
    def provide_input(self, data: UserInputData) -> None:
        self._user_input = data

    @workflow.run
    async def run(self, data: UserInputExampleWorkflowInput) -> dict:
        # 1. Generate a random number between the specified bounds
        random_number = await workflow.execute_activity(
            generate_random_number_activity,
            args=[data.lower_bound, data.upper_bound],
            start_to_close_timeout=timedelta(seconds=10),
        )

        # 2. Ask the backend to log that we are waiting for user input
        await request_signal_input(
            action_instance_id=workflow.info().workflow_id,
            user_id=data.user_id,
            signal_fn_name='provide_input',
            title='Review Required',
            description=(
                'Please approve or reject the randomly generated '
                f'number: {random_number}.'
            ),
        )

        # 3. Suspend workflow execution until the signal is received
        await workflow.wait_condition(lambda: self._user_input is not None)

        # 4. Resume and return the final decision regarding the random number
        decision_str = (
            'approved' if self._user_input.decision.lower() == 'approve' else 'rejected'
        )
        file_metadata = await workflow.execute_activity(
            collect_signal_asset_metadata_activity,
            args=[
                workflow.info().workflow_id,
                self._user_input.image_path,
                self._user_input.audio_submission,
            ],
            start_to_close_timeout=timedelta(seconds=10),
        )
        return {
            'status': 'success',
            'message': f'user {decision_str} {random_number}',
            'user_notes': self._user_input.notes,
            'submitted_file_metadata': file_metadata,
        }
