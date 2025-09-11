from dataclasses import dataclass
from typing import Literal


SpaceGroupLiteral = Literal[
    '',
    'P 1',
    'P -1',
    'P 2',
    'P 21',
    'C 2',
    'P m',
    'P c',
    'C m',
    'C c',
    'P 2/m',
    'P 21/m',
    'C 2/m',
    'P 2/c',
    'P 21/c',
    'C 2/c',
    'P 2 2 2',
    'P 2 2 21',
    'P 21 21 2',
    'P 21 21 21',
    'C 2 2 21',
    'C 2 2 2',
    'F 2 2 2',
    'I 2 2 2',
    'I 21 21 21',
    'P m m 2',
    'P m c 21',
    'P c c 2',
    'P m a 2',
    'P c a 21',
    'P n c 2',
    'P m n 21',
    'P b a 2',
    'P n a 21',
    'P n n 2',
    'C m m 2',
    'C m c 21',
    'C c c 2',
    'A m m 2',
    'A e m 2',
    'A m a 2',
    'A e a 2',
    'F m m 2',
    'F d d 2',
    'I m m 2',
    'I b a 2',
    'I m a 2',
    'P m m m',
    'P n n n',
    'P c c m',
    'P b a n',
    'P m m a',
    'P n n a',
    'P m n a',
    'P c c a',
    'P b a m',
    'P c c n',
    'P b c m',
    'P n n m',
    'P m m n',
    'P b c n',
    'P b c a',
    'P n m a',
    'C m c m',
    'C m c e',
    'C m m m',
    'C c c m',
    'C m m e',
    'C c c e',
    'F m m m',
    'F d d d',
    'I m m m',
    'I b a m',
    'I b c a',
    'I m m a',
    'P 4',
    'P 41',
    'P 42',
    'P 43',
    'I 4',
    'I 41',
    'P -4',
    'I -4',
    'P 4/m',
    'P 42/m',
    'P 4/n',
    'P 42/n',
    'I 4/m',
    'I 41/a',
    'P 4 2 2',
    'P 4 21 2',
    'P 41 2 2',
    'P 41 21 2',
    'P 42 2 2',
    'P 42 21 2',
    'P 43 2 2',
    'P 43 21 2',
    'I 4 2 2',
    'I 41 2 2',
    'P 4 m m',
    'P 4 b m',
    'P 42 c m',
    'P 42 n m',
    'P 4 c c',
    'P 4 n c',
    'P 42 m c',
    'P 42 b c',
    'I 4 m m',
    'I 4 c m',
    'I 41 m d',
    'I 41 c d',
    'P -4 2 m',
    'P -4 2 c',
    'P -4 21 m',
    'P -4 21 c',
    'P -4 m 2',
    'P -4 c 2',
    'P -4 b 2',
    'P -4 n 2',
    'I -4 m 2',
    'I -4 c 2',
    'I -4 2 m',
    'I -4 2 d',
    'P 4/m m m',
    'P 4/m c c',
    'P 4/n b m',
    'P 4/n n c',
    'P 4/m b m',
    'P 4/m n c',
    'P 4/n m m',
    'P 4/n c c',
    'P 42/m m c',
    'P 42/m c m',
    'P 42/n b c',
    'P 42/n n m',
    'P 42/m b c',
    'P 42/m n m',
    'P 42/n m c',
    'P 42/n c m',
    'I 4/m m m',
    'I 4/m c m',
    'I 41/a m d',
    'I 41/a c d',
    'P 3',
    'P 31',
    'P 32',
    'R 3',
    'P -3',
    'R -3',
    'P 3 1 2',
    'P 3 2 1',
    'P 31 1 2',
    'P 31 2 1',
    'P 32 1 2',
    'P 32 2 1',
    'R 3 2',
    'P 3 m 1',
    'P 3 1 m',
    'P 3 c 1',
    'P 3 1 c',
    'R 3 m',
    'R 3 c',
    'P -3 1 m',
    'P -3 1 c',
    'P -3 m 1',
    'P -3 c 1',
    'R -3 m',
    'R -3 c',
    'P 6',
    'P 61',
    'P 65',
    'P 62',
    'P 64',
    'P 63',
    'P -6',
    'P 6/m',
    'P 63/m',
    'P 6 2 2',
    'P 61 2 2',
    'P 65 2 2',
    'P 62 2 2',
    'P 64 2 2',
    'P 63 2 2',
    'P 6 m m',
    'P 6 c c',
    'P 63 c m',
    'P 63 m c',
    'P -6 m 2',
    'P -6 c 2',
    'P -6 2 m',
    'P -6 2 c',
    'P 6/m m m',
    'P 6/m c c',
    'P 63/m c m',
    'P 63/m m c',
    'P 2 3',
    'F 2 3',
    'I 2 3',
    'P 21 3',
    'I 21 3',
    'P m -3',
    'P n -3',
    'F m -3',
    'F d -3',
    'I m -3',
    'P a -3',
    'I a -3',
    'P 4 3 2',
    'P 42 3 2',
    'F 4 3 2',
    'F 41 3 2',
    'I 4 3 2',
    'P 43 3 2',
    'P 41 3 2',
    'I 41 3 2',
    'P -4 3 m',
    'F -4 3 m',
    'I -4 3 m',
    'P -4 3 n',
    'F -4 3 c',
    'I -4 3 d',
    'P m -3 m',
    'P n -3 n',
    'P m -3 n',
    'P n -3 m',
    'F m -3 m',
    'F m -3 c',
    'F d -3 m',
    'F d -3 c',
    'I m -3 m',
    'I a -3 d',
]

NumFormulaUnitsPerCell = Literal(['1', '2', '3', '4', '6', '8'])


@dataclass
class PromptGenerationInput:
    """
    Input data for the prompt generation workflow.

    Attributes:
    - input_composition: Composition to use as a prompt for the model.
    - input_num_formula_units_per_cell: Number of formula units per cell.
    - input_space_group: Space group to use in the prompt.
    """

    input_composition: str
    input_num_formula_units_per_cell: str = ''
    input_space_group: SpaceGroupLiteral = ''


@dataclass
class InferenceSettingsInput:
    model_path: str = 'models/crystallm_v1_small/ckpt.pt'
    model_url: str = (
        'https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz'
    )
    num_samples: int = 2
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 10
    seed: int = 1337
    dtype: str = 'bfloat16'
    compile: bool = False
    generate_cif: bool = True


@dataclass
class InferenceUserInput:
    """
    User input data for the inference workflow.

    Attributes:
    - input_composition: Composition to use as a prompt for the model.
    - input_num_formula_units_per_cell: Number of formula units per cell.
    - input_space_group: Space group to use in the prompt.
    - user_id: User making the request
    - upload_id: If `generate_cif` is set to True, save CIF files to this upload.
    - generate_cif: If True, the model will generate CIF files.
    """

    upload_id: str
    user_id: str
    prompt_generation_inputs: list[PromptGenerationInput]
    inference_settings: InferenceSettingsInput


@dataclass
class InferenceModelInput:
    """
    Model input data for the inference workflow.

    Attributes:

    - model_path: Path to the model file.
    - model_url: URL to download the model if not available locally.
    - raw_input: Raw input string to use as a prompt.
    - num_samples: Number of samples to draw during inference.
    - max_new_tokens: Maximum number of tokens to generate in each sample.
    - temperature: Controls the randomness of predictions. Lower values make the
        model more deterministic, while higher values increase randomness.
    - top_k: Retain only the top_k most likely tokens, clamp others to have 0
        probability.
    - seed: Random seed for reproducibility.
    - dtype: Data type for the model (e.g., 'float32', 'bfloat16', 'float16').
    - compile: Whether to compile the model for faster inference.
    """

    prompts: list[str]
    inference_settings: InferenceSettingsInput


@dataclass
class InferenceResultsInput:
    """
    CIF Results input data for the inference workflow.

    Attributes:
    - upload_id: If generate_cif, write the generate CIF files to the upload.
    - user_id: User making the request
    - generate_cif: If True, the model will generate CIF files.
    - generated_samples: List to store generated samples from the model.
    - cif_dir: Directory to save CIF files. If empty, uses the upload's raw directory.
    - cif_prefix: Prefix for the generated CIF files: <cif_prefix>_<index>.cif
    """

    upload_id: str
    user_id: str
    generated_samples: list[str]
    generate_cif: bool
    model_data: InferenceModelInput
    cif_dir: str = ''  # empty string means the upload's raw directory
    cif_prefix: str = 'sample'
