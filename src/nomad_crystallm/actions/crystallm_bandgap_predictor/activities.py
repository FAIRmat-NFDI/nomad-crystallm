import os

from nomad.actions.utils import get_upload_files
from temporalio import activity

from nomad_crystallm.actions.crystallm_bandgap_predictor.shared import (
    CIFDescriptionInput,
)


@activity.defn
async def cif_to_description(data: CIFDescriptionInput) -> str:
    """
    Converts a CIF file to a description string for the bandgap prediction model.
    """
    from pymatgen.core import Structure
    from robocrys import StructureCondenser, StructureDescriber

    upload_files = get_upload_files(data.upload_id, data.user_id)
    if upload_files is None:
        raise ValueError("Couldn't find upload files")
    cif_path = os.path.join(str(upload_files._raw_dir), data.cif_path)
    structure = Structure.from_file(cif_path)
    condenser = StructureCondenser()
    condensed_structure = condenser.condense_structure(structure)
    describer = StructureDescriber()
    description = describer.describe(condensed_structure)

    return str(description)
