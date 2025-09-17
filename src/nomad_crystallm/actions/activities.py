import pandas as pd
from nomad.datamodel import ServerContext
from temporalio import activity

from nomad_crystallm.actions.shared import (
    ConstructPromptInput,
    InferenceModelInput,
    InferenceResultsInput,
    PromptGenerationFileInput,
    PromptGenerationTextInput,
)
from nomad_crystallm.actions.utils import get_upload


@activity.defn
async def get_model(data: InferenceModelInput):
    from .llm import download_model

    await download_model(data.inference_settings.model_name)


@activity.defn
async def construct_prompts(
    data: ConstructPromptInput,
) -> list[str]:
    from .llm import construct_prompt

    prompts = []

    if isinstance(data.prompter, PromptGenerationTextInput):
        for prompt_generation_input in data.prompter.prompt_generation_inputs:
            prompts.append(
                construct_prompt(
                    prompt_generation_input.input_composition,
                    prompt_generation_input.input_num_formula_units_per_cell,
                    prompt_generation_input.input_space_group,
                )
            )
    elif isinstance(data.prompter, PromptGenerationFileInput):
        upload = get_upload(data.upload_id, data.user_id)
        context = ServerContext(upload)
        if not context.raw_path_exists(data.prompter.filepath):
            raise FileNotFoundError(
                f'File {data.prompter.filepath} not found in the raw folder of upload '
                f'{data.upload_id}.'
            )
        with context.raw_file(data.prompter.filepath) as f:
            df = pd.read_csv(f)

        required_columns = {
            'input_composition',
            'input_num_formula_units_per_cell',
            'input_space_group',
        }
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f'CSV file must contain the following columns: {required_columns}'
            )
        for _, row in df.iterrows():
            prompts.append(
                construct_prompt(
                    row['input_composition'],
                    row['input_num_formula_units_per_cell'],
                    row['input_space_group'],
                )
            )

    return prompts


@activity.defn
async def run_inference(data: InferenceModelInput) -> list[list[str]]:
    from .llm import evaluate_model

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
