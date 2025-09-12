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

    inference_results: list[list[str]]
    bandgap_predictions: list[BandGapPredictionOutput]


@dataclass
class CIFDescriptionInput:
    action_instance_id: str
    upload_id: str
    user_id: str


@dataclass
class WriteEntryInput:
    upload_id: str
    user_id: str
    prediction_outputs: list[BandGapPredictionOutput]
