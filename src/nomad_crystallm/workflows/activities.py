import os

from nomad.orchestrator.utils import workflow_artifacts_dir
from temporalio import activity

from nomad_crystallm.workflows.shared import (
    InferenceInput,
    InferenceModelInput,
    InferenceResultsInput,
)


@activity.defn
async def get_model(data: InferenceModelInput):
    from .llm import download_model

    model_path = os.path.join(workflow_artifacts_dir(), data.model_path)
    await download_model(model_path, data.model_url)


@activity.defn
async def construct_model_input(data: InferenceInput) -> str:
    from .llm import construct_prompt

    # validates that the composition is not empty
    if not data.input_composition:
        raise ValueError('Composition for the prompt cannot be empty.')
    # constructs the prompt for the model
    return construct_prompt(
        data.input_composition,
        data.input_num_formula_units_per_cell,
        data.input_space_group,
    )


@activity.defn
async def run_inference(data: InferenceModelInput) -> list[str]:
    from .llm import evaluate_model

    data.model_path = os.path.join(workflow_artifacts_dir(), data.model_path)
    return evaluate_model(data)


@activity.defn
async def write_results(data: InferenceResultsInput) -> None:
    """
    Write the inference results to a file.
    """
    from .llm import write_cif_files, write_entry_archive

    cif_paths = write_cif_files(data, logger=activity.logger)
    if not cif_paths:
        raise ValueError('No CIF files were generated.')
    write_entry_archive(cif_paths, data)
