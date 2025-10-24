from temporalio import activity

from nomad_crystallm.actions.inference.models import (
    InferenceInput,
    PromptConstructionInput,
    WriteResultsInput,
)


@activity.defn
async def get_model(model: str) -> None:
    from nomad_crystallm.actions.inference.utils import download_model

    await download_model(model)


@activity.defn
async def get_prompt(data: PromptConstructionInput) -> str:
    from nomad_crystallm.actions.inference.utils import construct_prompt

    return construct_prompt(
        data.composition,
        data.num_formula_units_per_cell,
        data.space_group,
    )


@activity.defn
async def run_inference(data: InferenceInput) -> list[str]:
    from nomad_crystallm.actions.inference.utils import evaluate_model

    return evaluate_model(data)


@activity.defn
async def write_results(data: WriteResultsInput) -> None:
    """
    Write the inference results to a file.
    """
    from nomad_crystallm.actions.inference.utils import (
        write_cif_files,
        write_entry_archive,
    )

    cif_paths = write_cif_files(data, logger=activity.logger)
    if not cif_paths:
        raise ValueError('No CIF files were generated.')
    write_entry_archive(cif_paths, data)
