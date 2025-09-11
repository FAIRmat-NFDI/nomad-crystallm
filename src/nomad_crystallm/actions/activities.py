import os

from nomad.actions.utils import action_artifacts_dir
from temporalio import activity

from nomad_crystallm.actions.shared import (
    InferenceModelInput,
    InferenceResultsInput,
    InferenceUserInput,
)


@activity.defn
async def get_model(data: InferenceModelInput):
    from .llm import download_model

    model_path = os.path.join(
        action_artifacts_dir(), data.inference_settings.model_path
    )
    await download_model(model_path, data.inference_settings.model_url)


@activity.defn
async def construct_model_input(data: InferenceUserInput) -> str:
    from .llm import construct_prompt

    # validates that the composition is not empty
    if not data.input_composition:
        raise ValueError('Composition for the prompt cannot be empty.')
    # constructs the prompt for the model
    prompts = []
    for prompt_generation_input in data.prompt_generation_inputs:
        prompts.append(
            construct_prompt(
                prompt_generation_input.input_composition,
                prompt_generation_input.input_num_formula_units_per_cell,
                prompt_generation_input.input_space_group,
            )
        )
    return prompts


@activity.defn
async def run_inference(data: InferenceModelInput) -> list[str]:
    from .llm import evaluate_model

    data.model_path = os.path.join(action_artifacts_dir(), data.model_path)
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
