from nomad.utils.structlogging import get_logger
from temporalio import activity

from nomad_crystallm.actions.inference.models import (
    InferenceInput,
    PromptConstructionInput,
    WriteResultsInput,
)

logger = get_logger('nomad_crystallm.actions.inference.activities')


@activity.defn
async def get_model(model: str) -> None:
    from nomad_crystallm.actions.inference.utils import download_model

    logger.info(f'Fetching model {model}...')

    await download_model(model)

    logger.info('Completed fetching model.')


@activity.defn
async def get_prompt(data: PromptConstructionInput) -> str:
    from nomad_crystallm.actions.inference.utils import construct_prompt

    logger.info('Constructing prompt for inference...')
    prompt = construct_prompt(
        data.composition,
        data.num_formula_units_per_cell,
        data.space_group,
    )
    logger.info('Completed constructing prompt.')
    return prompt


@activity.defn
async def run_inference(data: InferenceInput) -> list[str]:
    from nomad_crystallm.actions.inference.utils import evaluate_model

    logger.info('Running inference...')

    result = evaluate_model(data)

    logger.info('Completed running inference.')

    return result


@activity.defn
async def write_results(data: WriteResultsInput) -> None:
    """
    Write the inference results to a file.
    """
    from nomad_crystallm.actions.inference.utils import (
        write_cif_files,
        write_entry_archive,
    )

    logger.info('Writing inference results...')

    cif_paths = write_cif_files(data)
    if not cif_paths:
        logger.error('No CIF files were generated.')
        raise ValueError('No CIF files were generated.')
    write_entry_archive(cif_paths, data)

    logger.info('Completed writing inference results.')
