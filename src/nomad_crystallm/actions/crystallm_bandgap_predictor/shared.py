from dataclasses import dataclass

from nomad_crystallm.actions.bandgap_predictor.shared import BandGapPredictionOutput


@dataclass
class CrystaLLMBandGapPredictionOutput:
    """
    Output of the CrystaLLM + Bandgap prediction workflow.

    Attributes:
    - generated_samples: A list of generated samples.
    - bandgap_predictions: A list of bandgap prediction results.
    """

    generated_samples: list[str]
    bandgap_predictions: BandGapPredictionOutput


@dataclass
class CIFDescriptionInput:
    cif_path: str
    upload_id: str
    user_id: str
