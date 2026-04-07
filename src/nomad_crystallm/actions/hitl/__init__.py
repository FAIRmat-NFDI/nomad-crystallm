from nomad.actions import TaskQueue
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class MyActionEntryPoint(ActionEntryPoint):
    def load(self):
        from nomad.actions import Action

        from nomad_crystallm.actions.hitl.activities import (
            generate_random_number_activity,
        )
        from nomad_crystallm.actions.hitl.workflows import (
            UserInputExampleWorkflow,
        )

        return Action(
            task_queue=self.task_queue,
            workflow=UserInputExampleWorkflow,
            child_workflows=[],
            activities=[generate_random_number_activity],
        )


myaction = MyActionEntryPoint(
    name='MyAction',
    task_queue=TaskQueue.CPU,
    description='My custom action.',
)
