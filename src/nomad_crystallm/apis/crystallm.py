import uuid

from fastapi import Depends, FastAPI, HTTPException
from nomad.app import main
from nomad.app.v1.models import User
from nomad.app.v1.routers.auth import create_user_dependency
from nomad.config import config
from nomad.orchestrator.shared.constant import TaskQueue

from nomad_crystallm.workflows.shared import InferenceInput, InferenceUserInput

crystallm_api_entrypoint = config.get_plugin_entry_point(
    'nomad_crystallm.apis:crystallm_api'
)

root_path = f'{config.services.api_base_path}/{crystallm_api_entrypoint.prefix}'
print(root_path)

app = FastAPI(root_path=root_path)


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/start-inference-task')
async def start_inference_task(
    data: InferenceUserInput,
    user: User = Depends(create_user_dependency(required=True)),
):
    workflow_id = f'crystallm-workflow-{user.user_id}-{uuid.uuid4()}'
    client = main.temporal_client()
    workflow_data = InferenceInput(
        input_composition=data.input_composition,
        input_num_formula_units_per_cell=data.input_num_formula_units_per_cell,
        input_space_group=data.input_space_group,
        generate_cif=data.generate_cif,
        upload_id=data.upload_id,
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
    try:
        await client.start_workflow(
            'InferenceWorkflow', workflow_data, id=workflow_id, task_queue=TaskQueue.GPU
        )
        return {'workflow_id': workflow_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
