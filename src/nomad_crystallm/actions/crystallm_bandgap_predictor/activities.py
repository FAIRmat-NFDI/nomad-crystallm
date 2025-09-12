import os

from nomad.actions.utils import get_upload_files
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.processing import Upload
from temporalio import activity

from nomad_crystallm.actions.bandgap_predictor.shared import CIFDescriptionOutput
from nomad_crystallm.actions.crystallm_bandgap_predictor.shared import (
    CIFDescriptionInput,
    WriteEntryInput,
)


@activity.defn
async def cif_to_description(data: CIFDescriptionInput) -> list[CIFDescriptionOutput]:
    """
    Converts a CIF file to a description string for the bandgap prediction model.
    """
    from pymatgen.core import Structure
    from robocrys import StructureCondenser, StructureDescriber

    upload_files = get_upload_files(data.upload_id, data.user_id)
    if upload_files is None:
        raise ValueError("Couldn't find upload files")
    action_dir = os.path.join(str(upload_files._raw_dir), data.action_instance_id)
    composition_and_structures = []
    for composition_dir in os.listdir(action_dir):
        composition_entry_path = None
        structures = []
        for path in os.listdir(os.path.join(action_dir, composition_dir)):
            print(path)
            if path.endswith('archive.json'):
                composition_entry_path = os.path.join(action_dir, composition_dir, path)
            if path.endswith('.cif'):
                cif_path = os.path.join(action_dir, composition_dir, path)
                structures.append(Structure.from_file(cif_path))

        assert composition_entry_path is not None, 'Entry archive not found'
        composition_and_structures.append(
            {'composition_entry_path': composition_entry_path, 'structures': structures}
        )

    entrypath_and_descriptions: list[CIFDescriptionOutput] = []
    for composition_and_structure in composition_and_structures:
        descriptions = []
        for structure in composition_and_structure['structures']:
            condenser = StructureCondenser()
            condensed_structure = condenser.condense_structure(structure)
            describer = StructureDescriber()
            description = str(describer.describe(condensed_structure))
            descriptions.append(description)
        entrypath_and_descriptions.append(
            CIFDescriptionOutput(
                entry_path=str(composition_and_structure['composition_entry_path']),
                descriptions=descriptions,
            )
        )

    return entrypath_and_descriptions


@activity.defn
async def write_prediction_results(data: WriteEntryInput):
    """
    Write the inference results to a file.
    """
    import json

    upload = get_upload_with_read_access(
        data.upload_id,
        User(user_id=data.user_id),
        include_others=True,
    )

    for res in data.prediction_outputs:
        with open(res.entry_path, encoding='utf-8') as f:
            entry_data = json.loads(f.read())

        entry_data['data']['band_gap_prediction'] = [
            {
                'prediction': predict_result.prediction,
                'probability': predict_result.probability,
            }
            for predict_result in res.results
        ]

        with open(res.entry_path, 'w', encoding='utf-8') as f:
            json.dump(entry_data, f, indent=4)

        target_dir = os.path.dirname(res.entry_path).replace(
            f'{str(upload.upload_files._raw_dir)}/', ''
        )

        upload.process_upload(
            file_operations=[
                dict(
                    op='ADD',
                    path=res.entry_path,
                    target_dir=target_dir,
                    temporary=False,
                )
            ],
            only_updated_files=True,
        )
