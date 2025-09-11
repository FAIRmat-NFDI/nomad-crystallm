from dataclasses import dataclass


@dataclass
class BandGapPredictionInput:
    """
    Input for the bandgap prediction workflow.

    Attributes:
    - descriptions: A list of material descriptions (strings).
    """

    descriptions: list[str]
    upload_id: str
    user_id: str


@dataclass
class BandGapPredictionResult:
    """
    Result of a single bandgap prediction.

    Attributes:
    - prediction: The predicted class (0 or 1).
    - probability: The probability of the prediction.
    """

    prediction: bool
    probability: float


@dataclass
class BandGapPredictionOutput:
    """
    Output of the bandgap prediction workflow.

    Attributes:
    - results: A list of bandgap prediction results.
    """

    results: list[BandGapPredictionResult]
