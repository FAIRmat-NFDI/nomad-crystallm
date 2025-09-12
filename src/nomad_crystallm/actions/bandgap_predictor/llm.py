import numpy as np
import torch
from torch import nn

from .shared import (
    BandGapPredictionInput,
    BandGapPredictionOutput,
    BandGapPredictionResult,
)


class BandGapPredictor(nn.Module):
    def __init__(
        self,
        base_model,
        base_model_output_size,
        n_classes=1,
        drop_rate=0.1,
    ):
        super().__init__()
        D_in, D_out = base_model_output_size, n_classes
        self.model = base_model
        self.dropout = nn.Dropout(drop_rate)

        # instantiate a linear regressor
        self.linear_regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out),
        )

    def forward(self, input_ids, attention_masks):
        hidden_states = self.model(input_ids, attention_masks)
        last_hidden_state = (
            hidden_states.last_hidden_state
        )  # [batch_size, input_length, D_in]
        input_embedding = last_hidden_state[:, 0, :]
        outputs = self.linear_regressor(input_embedding)  # [batch_size, D_out]
        return input_embedding, outputs


def prepare_batch_inputs(tokenizer, descriptions_list, max_length):
    """
    Tokenizes a list of description strings and prepares it as a model-ready batch of tensors.
    """
    # Prepend the [CLS] token to each description in the list
    texts_with_cls = ['[CLS] ' + str(desc) for desc in descriptions_list]

    encoded_batch = tokenizer(
        text=texts_with_cls,
        add_special_tokens=True,
        padding='longest',  # Pad to the length of the longest sequence in the batch
        truncation=True,  # Truncate sequences longer than max_length
        max_length=max_length,
        return_tensors='pt',  # Return PyTorch tensors
    )
    return encoded_batch


def predict_batch_classification(
    model, tokenizer, descriptions_list, device, max_length, threshold=0.5
):
    """
    Performs multi-class classification prediction on a batch of description strings.
    """
    # 1. Set the model to evaluation mode
    model.eval()

    # 2. Tokenize the batch of descriptions
    inputs = prepare_batch_inputs(tokenizer, descriptions_list, max_length)

    # 3. Move tensors to the correct device
    batch_inputs = inputs['input_ids'].to(device)
    batch_masks = inputs['attention_mask'].to(device)

    # 4. Perform prediction
    with torch.no_grad():
        # Get raw logits. Shape: [batch_size, 1]
        _, logits = model(batch_inputs, batch_masks)

        # Apply sigmoid to convert logits to probabilities of the positive class
        probabilities = torch.sigmoid(logits)

        # Apply threshold to get predicted class (0 or 1)
        predicted_classes = (probabilities > threshold).long().squeeze()

    # Move to CPU and convert to NumPy
    predicted_classes_cpu = np.atleast_1d(predicted_classes.cpu().numpy())
    probabilities_cpu = np.atleast_1d(probabilities.cpu().numpy().flatten())

    return predicted_classes_cpu, probabilities_cpu


def run_prediction(
    model,
    tokenizer,
    input_data: BandGapPredictionInput,
    device='cpu',
    max_length=888,
) -> list[BandGapPredictionOutput]:
    """
    Runs the prediction on a list of descriptions.
    """
    results = []
    for output in input_data.description_output:
        predictions, probabilities = predict_batch_classification(
            model, tokenizer, output.descriptions, device, max_length
        )
        entry_results = [
            BandGapPredictionResult(prediction=bool(p), probability=float(f))
            for p, f in zip(predictions, probabilities)
        ]
        results.append(
            BandGapPredictionOutput(results=entry_results, entry_path=output.entry_path)
        )
    return results
