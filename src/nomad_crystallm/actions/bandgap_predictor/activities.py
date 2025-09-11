import asyncio
import os
import shutil
import tarfile
import tempfile

from nomad.actions.utils import action_artifacts_dir
from temporalio import activity

from .shared import BandGapPredictionInput, BandGapPredictionOutput

BLOCK_SIZE = 1024


@activity.defn
async def setup_model(url: str) -> str:
    """
    Downloads and decompresses the model checkpoint.
    """
    import aiohttp

    model_path = 'best_checkpoint_for_is_gap_direct.pt'
    model_path = os.path.join(action_artifacts_dir(), model_path)
    exists = await asyncio.to_thread(os.path.exists, model_path)
    if exists:
        return model_path
    with tempfile.TemporaryDirectory() as tmpdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                # Download in chunks
                tmp_zipfile = os.path.join(tmpdir, url.split('/')[-1])
                loop = asyncio.get_running_loop()
                with open(tmp_zipfile, 'wb') as f:
                    async for chunk in response.content.iter_chunked(BLOCK_SIZE):
                        await loop.run_in_executor(None, f.write, chunk)
        # Unpack the model zip
        with tarfile.open(tmp_zipfile, 'r:gz') as tar:
            tar.extractall(tmpdir)
        tmp_zipdir = os.path.join(tmpdir, 'checkpoints', 'samples', 'classification')
        # Check if '.pt' file exists in the extracted directory
        model_files = [f for f in os.listdir(tmp_zipdir) if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(
                'No ".pt" file found in the extracted directory '
                f'"{os.path.dirname(model_path)}".'
            )
        # Move over the first .pt file found to the model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.move(os.path.join(tmp_zipdir, model_files[0]), model_path)
    return model_path


@activity.defn
async def run_prediction_activity(
    model_path: str, input_data: 'BandGapPredictionInput'
) -> 'BandGapPredictionOutput':
    """
    Runs the prediction activity.
    """
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    from .llm import BandGapPredictor, run_prediction

    base_model = T5EncoderModel.from_pretrained('google/t5-v1_1-small')
    tokenizer_path = Path(__file__).parent.resolve() / 'tokenizers'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_tokens(['[CLS]'])
    tokenizer.add_tokens(['[NUM]'])  # special token to replace bond lengths
    tokenizer.add_tokens(['[ANG]'])  # special token to replace bond angles

    base_model.resize_token_embeddings(len(tokenizer))

    for param in base_model.parameters():
        param.requires_grad = False

    drop_rate = 0.2
    base_model_output_size = 512
    model = BandGapPredictor(base_model, base_model_output_size, drop_rate=drop_rate)
    model.load_state_dict(torch.load(model_path), strict=False)

    return run_prediction(model, tokenizer, input_data)
