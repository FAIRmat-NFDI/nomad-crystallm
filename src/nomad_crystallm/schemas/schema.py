import os

from ase.io import read
from ase.spacegroup import Spacegroup
from matid import SymmetryAnalyzer
from nomad.datamodel.data import ArchiveSection, EntryData, EntryDataCategory
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    ELNComponentEnum,
    SectionProperties,
)
from nomad.datamodel.results import Material, Results, SymmetryNew, System
from nomad.metainfo import Category, MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad.normalizing.common import nomad_atoms_from_ase_atoms
from nomad.normalizing.topology import add_system, add_system_info
from nomad.orchestrator import utils as orchestrator_utils
from nomad.orchestrator.shared.constant import TaskQueue
from pymatgen.core import Composition

from nomad_crystallm.schemas.utils import get_reference_from_mainfile
from nomad_crystallm.workflows.shared import InferenceUserInput

SPACE_GROUPS = [Spacegroup(i).symbol for i in range(1, 231)]

m_package = SchemaPackage()


class InferenceCategory(EntryDataCategory):
    """Category for inference workflows."""

    m_def = Category(label='Inference Workflows', categories=[EntryDataCategory])


class RunWorkflowAction(ArchiveSection):
    """Abstract section to run inference workflows"""

    trigger_run_workflow = Quantity(
        type=bool,
        description='Starts an asynchronous workflow for running the inference.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Run Inference Workflow',
        ),
    )

    def run_workflow(self, archive, logger=None):
        """Run the workflow with the provided archive."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        if self.trigger_run_workflow:
            try:
                self.run_workflow(archive, logger)
            except Exception as e:
                logger.error(f'Error running workflow: {e}. ')
            self.trigger_run_workflow = False


class InferenceSettings(ArchiveSection):
    """Settings for CrystaLLM inference workflows."""

    model = Quantity(
        type=MEnum(
            [
                'crystallm_v1_small',
                'crystallm_v1_large',
            ]
        ),
        description="""
        Model used for inference.
        | Available models                                 | Description                                                                         |
        |--------------------------------------------------|-------------------------------------------------------------------------------------|
        | **crystallm_v1_small  (25.36M parameters)**      | Downloadable at https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz |
        | **crystallm_v1_large  (1.2B parameters)**        | Downloadable at https://zenodo.org/records/10642388/files/crystallm_v1_large.tar.gz |
        """,  # noqa: E501
    )
    num_samples = Quantity(
        type=int,
        description='Number of samples to draw during inference.',
    )
    max_new_tokens = Quantity(
        type=int,
        description='Maximum number of tokens to generate in each sample.',
    )
    temperature = Quantity(
        type=float,
        description='Controls the randomness of predictions. Lower values make the '
        'model more deterministic, while higher values increase randomness.',
    )
    top_k = Quantity(
        type=int,
        description='Retain only the top_k most likely tokens, clamp others to have 0 '
        'probability.',
    )
    seed = Quantity(
        type=int,
        description='Random seed for reproducibility.',
    )
    dtype = Quantity(
        type=MEnum(['float32', 'bfloat16', 'float16']),
        description='Data type for the model (e.g., "float32", "bfloat16", "float16").',
    )
    compile = Quantity(
        type=bool,
        description='Whether to compile the model for faster inference.',
    )


class CrystaLLMInferenceResult(EntryData):
    """Result of a CrystaLLM inference workflow."""

    m_def = Section(
        label='CrystaLLM Inference Result',
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'prompt',
                    'workflow_id',
                    'status',
                    'trigger_workflow_status',
                    'generated_cifs',
                    'generated_structures',
                    'inference_settings',
                ]
            ),
        ),
    )
    prompt = Quantity(
        type=str,
        description='Prompt to be used for inference.',
    )
    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow.',
    )
    generated_cifs = Quantity(
        type=str,
        shape=['*'],
        description='Path to the CIF generated by the LLM.',
    )
    generated_structures = Quantity(
        type=System,
        shape=['*'],
        description='Reference to the system normalized based on the generated CIF.',
    )
    inference_settings = SubSection(
        section_def=InferenceSettings,
        description='Settings used for the CrystaLLM inference workflow.',
    )

    def process_generated_cifs(self, archive, logger):
        """
        Process the generated CIFs in `archive.data` and populates
        `archive.results.material`.
        """
        if not self.generated_cifs:
            return
        elements = set()
        topologies = {}
        system_references = []
        for cif in self.generated_cifs:
            if not cif:
                continue
            with archive.m_context.raw_file(cif) as file:
                try:
                    ase_atoms = read(file.name)
                except Exception as e:
                    logger.error(f'Unable to read cif: {cif}. Encounter the error: {e}')
                    continue
            # populate elemets from a set of all the elemsts in ase_atoms
            elements.update(ase_atoms.get_chemical_symbols())
            symmetry = SymmetryNew()
            symmetry_analyzer = SymmetryAnalyzer(ase_atoms, symmetry_tol=1)
            symmetry.bravais_lattice = symmetry_analyzer.get_bravais_lattice()
            symmetry.space_group_number = symmetry_analyzer.get_space_group_number()
            symmetry.space_group_symbol = (
                symmetry_analyzer.get_space_group_international_short()
            )
            symmetry.crystal_system = symmetry_analyzer.get_crystal_system()
            symmetry.point_group = symmetry_analyzer.get_point_group()

            label = f'{ase_atoms.get_chemical_formula()}-{symmetry.space_group_number}'
            system = System(
                atoms=nomad_atoms_from_ase_atoms(ase_atoms),
                label=label,
                description='Structure generated by CrystaLLM with workflow_id: '
                f'"{self.workflow_id}"',
                structural_type='bulk',
                dimensionality='3D',
                symmetry=symmetry,
            )
            add_system_info(system, topologies)
            add_system(system, topologies)

            # `system.system_id` is same as the reference to the topology index
            # `#/results/material/topology/{topology_iter}`
            system_references.append(f'#/{system.system_id}')
        self.generated_structures = system_references

        # reset results
        archive.results = Results(material=Material())
        archive.results.material.elements = list(elements)
        archive.results.material.topology = list(topologies.values())

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        self.process_generated_cifs(archive, logger)

        super().normalize(archive, logger)


class InferenceSettingsForm(ArchiveSection):
    """Settings used for CrystaLLM inference workflows with editable fields."""

    model = InferenceSettings.model.m_copy(deep=True)
    model.default = 'crystallm_v1_small'
    model.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.EnumEditQuantity,
    )

    num_samples = InferenceSettings.num_samples.m_copy(deep=True)
    num_samples.default = 1
    num_samples.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    max_new_tokens = InferenceSettings.max_new_tokens.m_copy(deep=True)
    max_new_tokens.default = 3000
    max_new_tokens.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    temperature = InferenceSettings.temperature.m_copy(deep=True)
    temperature.default = 0.8
    temperature.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    top_k = InferenceSettings.top_k.m_copy(deep=True)
    top_k.default = 10
    top_k.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    seed = InferenceSettings.seed.m_copy(deep=True)
    seed.default = 1337
    seed.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.NumberEditQuantity,
    )

    dtype = InferenceSettings.dtype.m_copy(deep=True)
    dtype.default = 'bfloat16'
    dtype.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.EnumEditQuantity,
    )

    compile = InferenceSettings.compile.m_copy(deep=True)
    compile.default = False
    compile.m_annotations['eln'] = ELNAnnotation(
        component=ELNComponentEnum.BoolEditQuantity,
    )


class InferenceStatus(ArchiveSection):
    """Section to fetch the status of an inference workflow."""

    workflow_id = Quantity(
        type=str,
        description='ID of the `temporalio` workflow.',
    )
    status = Quantity(
        type=str,
        description='Status of the inference workflow.',
    )
    generated_entry = Quantity(
        type=CrystaLLMInferenceResult,
        description='Reference to the generated entry after the workflow completes.',
    )
    trigger_get_status = Quantity(
        type=bool,
        default=False,
        description='Retrieve the current status of the inference workflow.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Get Workflow Status',
        ),
    )

    def normalize(self, archive, logger=None):
        """Normalize the section to ensure it is ready for processing."""
        super().normalize(archive, logger)
        if not self.status or self.status == 'RUNNING' or self.trigger_get_status:
            try:
                status = orchestrator_utils.get_workflow_status(self.workflow_id)
                if status:
                    self.status = status.name
            except Exception as e:
                logger.error(f'Error getting workflow status: {e}. ')
            finally:
                self.trigger_get_status = False
            if self.status == 'COMPLETED':
                reference = get_reference_from_mainfile(
                    archive.metadata.upload_id,
                    os.path.join(self.workflow_id, 'inference_result.archive.json'),
                )
                if not reference:
                    logger.error(
                        'Unable to set reference for the generated entry for '
                        f'workflow {self.workflow_id}.'
                    )
                else:
                    self.generated_entry = reference


class CrystaLLMInferenceForm(RunWorkflowAction, EntryData):
    """Inference form for running CrystaLLM inference workflows."""

    m_def = Section(
        a_eln=ELNAnnotation(
            label='CrystaLLM Inference Form',
        )
    )

    # TODO: Add field to upload a file containing multiple prompts.
    composition = Quantity(
        type=str,
        description='Chemical composition to be used for prompt. '
        'For example, NaCl, Al2O3.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    num_formula_units_per_cell = Quantity(
        type=MEnum(['1', '2', '3', '4', '6', '8']),
        description='(Optional) Number of formula units per unit cell to be used for '
        'prompt.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.AutocompleteEditQuantity),
    )
    space_group = Quantity(
        type=MEnum(SPACE_GROUPS),
        description='(Optional) Space group to be used for prompt.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.AutocompleteEditQuantity),
    )
    inference_settings = SubSection(
        section_def=InferenceSettingsForm,
        description='Settings for the CrystaLLM inference workflow.',
    )
    triggered_inferences = SubSection(
        section_def=InferenceStatus,
        description='A section for storing the status of the triggered inference '
        'workflow.',
        repeats=True,
    )

    def run_workflow(self, archive, logger=None):
        """
        Run the CrystaLLM inference workflow with the provided archive.
        Uses the first author's credentials to run the workflow.
        """
        if not self.composition:
            logger.warn(
                'No composition provided for the CrystaLLM inference prompt. '
                'Cannot run the workflow.'
            )
            return
        try:
            Composition(self.composition, strict=True)
        except Exception as e:
            logger.error(f'Invalid composition "{self.composition}": {e}')
            return
        if not archive.metadata.authors:
            logger.warn(
                'No authors found in the archive metadata. '
                'Cannot run CrystaLLM inference workflow.'
            )
            return
        if not self.inference_settings:
            self.inference_settings = InferenceSettingsForm()
        input_data = InferenceUserInput(
            user_id=archive.metadata.authors[0].user_id,
            upload_id=archive.metadata.upload_id,
            input_composition=self.composition,
            input_num_formula_units_per_cell=self.num_formula_units_per_cell or '',
            input_space_group=self.space_group or '',
            generate_cif=True,
            model_path=f'models/{self.inference_settings.model}/ckpt.pt',
            model_url=(
                'https://zenodo.org/records/10642388/files/'
                f'{self.inference_settings.model}.tar.gz'
            ),
            num_samples=self.inference_settings.num_samples,
            max_new_tokens=self.inference_settings.max_new_tokens,
            temperature=self.inference_settings.temperature,
            top_k=self.inference_settings.top_k,
            seed=self.inference_settings.seed,
            dtype=self.inference_settings.dtype,
            compile=self.inference_settings.compile,
        )
        workflow_name = 'nomad_crystallm.workflows.InferenceWorkflow'
        workflow_id = orchestrator_utils.start_workflow(
            workflow_name=workflow_name, data=input_data, task_queue=TaskQueue.GPU
        )
        if not self.triggered_inferences:
            self.triggered_inferences = [InferenceStatus()]
        else:
            self.triggered_inferences.append(InferenceStatus())

        self.triggered_inferences[-1].workflow_id = workflow_id

    def normalize(self, archive, logger=None):
        """
        Normalize the CrystaLLM inference form section.
        This method ensures that the section is ready for processing.
        """
        self.m_setdefault('inference_settings')
        super().normalize(archive, logger)
