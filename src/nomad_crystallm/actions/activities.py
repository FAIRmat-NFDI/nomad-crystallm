
from temporalio import activity

from nomad_crystallm.actions.shared import (
    InferenceModelInput,
    InferenceResultsInput,
    PromptGenerationInput,
)


@activity.defn
async def get_model(data: InferenceModelInput):
    from .llm import download_model

    await download_model(data.inference_settings.model_name)


@activity.defn
async def construct_model_input(
    prompt_generation_inputs: list[PromptGenerationInput],
) -> list[str]:
    from .llm import construct_prompt

    prompts = []

    # constructs the prompt for the model
    for prompt_generation_input in prompt_generation_inputs:
        # validates that the composition is not empty
        if not prompt_generation_input.input_composition:
            raise ValueError('Composition for the prompt cannot be empty.')
        prompts.append(
            construct_prompt(
                prompt_generation_input.input_composition,
                prompt_generation_input.input_num_formula_units_per_cell,
                prompt_generation_input.input_space_group,
            )
        )

    return prompts


@activity.defn
async def run_inference(data: InferenceModelInput) -> list[list[str]]:
    from .llm import evaluate_model

    return evaluate_model(data)


@activity.defn
async def write_results(data: InferenceResultsInput) -> list[str]:
    """
    Write the inference results to a file.
    """
    from .llm import write_cif_files, write_entry_archive

    cif_paths = write_cif_files(data, logger=activity.logger)
    if not cif_paths:
        raise ValueError('No CIF files were generated.')
    write_entry_archive(cif_paths, data)
    return cif_paths
