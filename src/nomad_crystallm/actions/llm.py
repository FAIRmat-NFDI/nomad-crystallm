import asyncio
import json
import os
import shutil
import tarfile
import tempfile
from contextlib import nullcontext
from typing import TYPE_CHECKING

import aiohttp
import torch
from crystallm import (
    GPT,
    CIFTokenizer,
    GPTConfig,
    extract_space_group_symbol,
    get_atomic_props_block_for_formula,
    remove_atom_props_block,
    replace_symmetry_operators,
)
from nomad.actions.utils import action_artifacts_dir, get_upload_files
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from pymatgen.core import Composition

from nomad_crystallm.actions.shared import (
    InferenceModelInput,
    InferenceResultsInput,
)
from nomad_crystallm.schemas.schema import (
    CrystaLLMInferenceResult,
    InferenceSettings,
)

if TYPE_CHECKING:
    from logging import LoggerAdapter
BLOCK_SIZE = 1024

model_data = {
    'crystallm_v1_small': {
        'model_path': 'models/crystallm_v1_small/ckpt.pt',
        'model_url': (
            'https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz'
        ),
    },
    'crystallm_v1_large': {
        'model_path': 'models/crystallm_v1_large/ckpt.pt',
        'model_url': (
            'https://zenodo.org/records/10642388/files/crystallm_v1_large.tar.gz'
        ),
    },
}


async def download_model(model_name: str) -> dict:
    """
    Checks if the model file exists locally, and if not, downloads it from the
    provided URL.
    """
    model_path = model_data[model_name]['model_path']
    model_path = os.path.join(action_artifacts_dir(), model_path)

    model_url = model_data[model_name]['model_url']

    # Check if file exists asynchronously
    exists = await asyncio.to_thread(os.path.exists, model_path)
    if not exists and not model_url:
        raise FileNotFoundError(
            f'Model file "{model_path}" does not exist and `model_url` is not provided.'
        )
    elif exists and model_url:
        return {
            'model_path': model_path,
            'model_url': model_url,
        }
    elif exists:
        return {'model_path': model_path}

    # Download the model from the URL and copy the model file to the model_path
    with tempfile.TemporaryDirectory() as tmpdir:
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                response.raise_for_status()
                # Download in chunks
                tmp_zipfile = os.path.join(tmpdir, model_url.split('/')[-1])
                loop = asyncio.get_running_loop()
                with open(tmp_zipfile, 'wb') as f:
                    async for chunk in response.content.iter_chunked(BLOCK_SIZE):
                        await loop.run_in_executor(None, f.write, chunk)
        # Unpack the model zip
        with tarfile.open(tmp_zipfile, 'r:gz') as tar:
            tar.extractall(tmpdir)
        tmp_zipdir = tmp_zipfile.split('.')[0]
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

    return {'model_path': model_path, 'model_url': model_url}


def construct_prompt(
    composition: str,
    num_formula_units_per_cell: str,
    space_group: str,
) -> str:
    """
    Construct the prompt for CrystaLLM inference based on the provided
    composition, number of formula units per cell, and space group.
    """
    # replace the factor with provided number of formula units per cell
    comp = Composition(composition)
    reduced_comp, factor = comp.get_reduced_composition_and_factor()
    if num_formula_units_per_cell:
        factor = int(num_formula_units_per_cell)
    comp_with_provided_factor = reduced_comp * factor
    comp_with_provided_factor_str = ''.join(comp_with_provided_factor.formula.split())

    if space_group:
        space_group_str = ''.join(space_group.split())
        return (
            f'data_{comp_with_provided_factor_str}\n'
            f'{get_atomic_props_block_for_formula(comp_with_provided_factor_str)}\n'
            f'_symmetry_space_group_name_H-M {space_group_str}\n'
        )

    return f'data_{comp_with_provided_factor_str}\n'


def evaluate_model(inference_state: InferenceModelInput) -> list[list[str]]:
    """
    Evaluate the model with the given parameters.
    Adapted from https://github.com/lantunes/CrystaLLM

    Returns a list of generated CIF strings for all the compositions.
    """
    torch.manual_seed(inference_state.inference_settings.seed)
    torch.cuda.manual_seed(inference_state.inference_settings.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device = (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )  # for later use in torch.autocast
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[inference_state.inference_settings.dtype]
    ctx = (
        nullcontext()
        if device == 'cpu'
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    model_path = model_data[inference_state.inference_settings.model_name]['model_path']
    model_path = os.path.join(action_artifacts_dir(), model_path)
    checkpoint = torch.load(model_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if inference_state.inference_settings.compile:
        model = torch.compile(model)

    # run generation
    all_generated = []
    for prompt in inference_state.prompts:
        # encode the beginning of the prompt
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        generated = []
        with torch.no_grad():
            with ctx:
                for k in range(inference_state.inference_settings.num_samples):
                    y = model.generate(
                        x,
                        inference_state.inference_settings.max_new_tokens,
                        temperature=inference_state.inference_settings.temperature,
                        top_k=inference_state.inference_settings.top_k,
                    )
                    generated.append(decode(y[0].tolist()))
        all_generated.append(generated)

    return all_generated


def postprocess(cif: str, fname: str, logger: 'LoggerAdapter') -> str:
    """
    Post-process the CIF file to ensure it is in a valid format.
    """
    try:
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != 'P 1':
            cif = replace_symmetry_operators(cif, space_group_symbol)

        # remove atom props
        cif = remove_atom_props_block(cif)
    except Exception as e:
        cif = '# WARNING: CrystaLLM could not post-process this file properly!\n' + cif
        logger.error(
            f"Error post-processing CIF file '{fname}': {e}",
            exc_info=True,
        )

    return cif


def write_cif_files(
    result: InferenceResultsInput, logger: 'LoggerAdapter'
) -> list[str]:
    """
    Write the generated CIFs to the specified target (console or file).
    """
    if not result.generate_cif:
        return []
    upload_files = get_upload_files(result.upload_id, result.user_id)
    if not upload_files:
        raise ValueError(
            f'No upload files found for upload_id "{result.upload_id}" '
            f'and user_id "{result.user_id}".'
        )
    cif_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for k, sample in enumerate(result.generated_samples):
            fname = os.path.join(tmpdir, f'{result.composition}_{k + 1}.cif')
            processed_sample = postprocess(sample, fname, logger)
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(processed_sample)
            cif_dir = os.path.join(result.action_instance_id, result.relative_cif_dir)
            upload_files.add_rawfiles(fname, target_dir=cif_dir)
            cif_paths.append(os.path.join(cif_dir, f'{result.composition}_{k + 1}.cif'))
    return cif_paths


def write_entry_archive(cif_paths, result: InferenceResultsInput) -> str:
    """
    Create an entry for the inference results and add it to the upload.
    """

    # upload_files = get_upload_files(result.upload_id, result.user_id)
    upload = get_upload_with_read_access(
        result.upload_id,
        User(user_id=result.user_id),
        include_others=True,
    )
    inference_result = CrystaLLMInferenceResult(
        prompt=result.prompt,
        action_instance_id=result.action_instance_id,
        generated_cifs=cif_paths,
        inference_settings=InferenceSettings(
            model=result.inference_settings.model_name,
            num_samples=result.inference_settings.num_samples,
            max_new_tokens=result.inference_settings.max_new_tokens,
            temperature=result.inference_settings.temperature,
            top_k=result.inference_settings.top_k,
            seed=result.inference_settings.seed,
            dtype=result.inference_settings.dtype,
            compile=result.inference_settings.compile,
        ),
    )
    fname = os.path.join(f'crystallm_{result.composition}.archive.json')
    with open(os.path.join(fname), 'w', encoding='utf-8') as f:
        json.dump({'data': inference_result.m_to_dict(with_root_def=True)}, f, indent=4)
    upload.process_upload(
        file_operations=[
            dict(
                op='ADD',
                path=fname,
                target_dir=os.path.join(
                    result.action_instance_id, result.relative_cif_dir
                ),
                temporary=True,
            )
        ],
        only_updated_files=True,
    )
