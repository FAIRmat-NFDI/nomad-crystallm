from collections import Counter
from unittest.mock import Mock

import pytest

from nomad_crystallm.actions.inference.models import (
    InferenceInput,
    InferenceSettingsInput,
)


@pytest.fixture
def inference_utils(monkeypatch):
    from nomad_crystallm.actions.inference import utils

    # Keep the regression check on CPU even on workers with CUDA installed.
    monkeypatch.setattr(utils.torch.cuda, 'is_available', lambda: False)
    return utils


def make_input(prompts, **settings):
    return InferenceInput(
        user_id='test-user',
        upload_id='test-upload',
        prompts=prompts,
        inference_settings=InferenceSettingsInput(**settings),
    )


def prompt_examples(utils):
    return [
        utils.construct_prompt('TiO2', '2', 'P 1'),
        utils.construct_prompt('NaCl', '1', ''),
        utils.construct_prompt('Si', '8', 'F d -3 m'),
        utils.construct_prompt('KCl', '1', ''),
        utils.construct_prompt('NaCl', '1', ''),
    ]


@pytest.mark.parametrize('scenario', ['mixed', 'uniform', 'single'])
@pytest.mark.parametrize('compile_model', [False, True])
def test_generation_batches_compatible_prompts_and_restores_order(
    inference_utils, monkeypatch, scenario, compile_model
):
    utils = inference_utils
    prompts = prompt_examples(utils)
    if scenario == 'uniform':
        prompts = [prompts[1], prompts[3], prompts[4]]
    elif scenario == 'single':
        prompts = [prompts[0]]
    tokenizer = utils.CIFTokenizer()
    encoded = [tokenizer.encode(tokenizer.tokenize_cif(prompt)) for prompt in prompts]
    expected_group_sizes = Counter(len(tokens) for tokens in encoded)
    if scenario == 'mixed':
        assert len(expected_group_sizes) > 1
    else:
        assert len(expected_group_sizes) == 1

    model = Mock()
    model.generate.side_effect = lambda tensor, *_, **__: tensor
    # Use the real postprocessor and real token tensors, without a large checkpoint.
    postprocess = utils.GPT.postprocess_generation
    model.postprocess_generation.side_effect = lambda tensor: postprocess(model, tensor)
    factory = Mock(return_value=model)
    monkeypatch.setattr(utils, 'GPT', factory)
    monkeypatch.setattr(utils, 'GPTConfig', Mock())
    monkeypatch.setattr(
        utils.torch, 'load', Mock(return_value={'model_args': {}, 'model': {}})
    )
    compile_mock = Mock(return_value=model)
    monkeypatch.setattr(utils.torch, 'compile', compile_mock)

    result = utils.evaluate_model(make_input(prompts, compile=compile_model))

    assert result == [
        tokenizer.decode(tokens).strip('\n') + '\n\n' for tokens in encoded
    ]
    assert model.generate.call_count == len(expected_group_sizes)
    actual_group_sizes = {}
    actual_rows = []
    for call in model.generate.call_args_list:
        tensor = call.args[0]
        batch_size, token_count = tensor.shape
        assert tensor.dtype == utils.torch.long
        actual_group_sizes[token_count] = batch_size
        actual_rows.extend(tensor.tolist())
    assert actual_group_sizes == expected_group_sizes
    assert Counter(map(tuple, actual_rows)) == Counter(map(tuple, encoded))
    factory.assert_called_once()
    assert compile_mock.call_count == int(compile_model)


def test_mixed_prompt_lengths_with_real_cpu_generation(
    inference_utils, monkeypatch, tmp_path
):
    utils = inference_utils
    tokenizer = utils.CIFTokenizer()
    config = utils.GPTConfig(
        vocab_size=len(tokenizer.token_to_id),
        block_size=128,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0,
    )
    model = utils.GPT(config)
    checkpoint = tmp_path / utils.model_data['crystallm_v1_small']['model_path']
    checkpoint.parent.mkdir(parents=True)
    utils.torch.save(
        {'model_args': vars(config), 'model': model.state_dict()}, checkpoint
    )
    monkeypatch.setattr(utils, 'action_artifacts_dir', lambda: str(tmp_path))
    prompts = prompt_examples(utils)

    result = utils.evaluate_model(
        make_input(prompts, max_new_tokens=2, dtype='float32')
    )

    assert len(result) == len(prompts)
    for prompt, sample in zip(prompts, result, strict=True):
        normalized_prompt = tokenizer.decode(
            tokenizer.encode(tokenizer.tokenize_cif(prompt))
        )
        assert sample.startswith(normalized_prompt)
        assert sample.endswith('\n\n')
