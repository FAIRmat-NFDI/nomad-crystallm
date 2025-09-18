from temporalio import activity

from nomad_crystallm.actions.shared import (
    ConstructPromptInput,
    ConstructPromptOutput,
    InferenceInput,
    InferenceOutput,
    PromptGenerationFileInput,
    PromptGenerationTextInput,
    WriteResultsInput,
)
from nomad_crystallm.actions.utils import get_upload


@activity.defn
async def get_model(model_name):
    from .llm import download_model

    await download_model(model_name)


@activity.defn
async def construct_prompts(
    data: ConstructPromptInput,
) -> list[ConstructPromptOutput]:
    import pandas as pd
    from nomad.datamodel import ServerContext

    from .llm import construct_prompt

    outputs = []

    if isinstance(data.prompter, PromptGenerationTextInput):
        for prompt_generation_input in data.prompter.prompt_generation_inputs:
            outputs.append(
                ConstructPromptOutput(
                    prompt=construct_prompt(
                        prompt_generation_input.input_composition,
                        prompt_generation_input.input_num_formula_units_per_cell,
                        prompt_generation_input.input_space_group,
                    ),
                    composition=prompt_generation_input.input_composition,
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
            outputs.append(
                ConstructPromptOutput(
                    prompt=construct_prompt(
                        str(row['input_composition'])
                        if not pd.isna(row['input_composition'])
                        else '',
                        str(int(row['input_num_formula_units_per_cell']))
                        if not pd.isna(row['input_num_formula_units_per_cell'])
                        else '',
                        str(row['input_space_group'])
                        if not pd.isna(row['input_space_group'])
                        else '',
                    ),
                    composition=str(row['input_composition']),
                )
            )

    return outputs


@activity.defn
async def run_inference(data: InferenceInput) -> InferenceOutput:
    from .llm import evaluate_model

    return evaluate_model(data)


@activity.defn
async def write_results(data: WriteResultsInput) -> None:
    """
    Write the inference results to a file.
    """
    from .llm import write_cif_files, write_entry_archive

    cif_paths = write_cif_files(data, logger=activity.logger)
    if not cif_paths:
        raise ValueError('No CIF files were generated.')
    write_entry_archive(cif_paths, data)
