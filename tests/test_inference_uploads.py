from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nomad.actions.assets.models import ActionAssetRef
from temporalio import workflow
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from nomad_crystallm.actions.inference import activities as activity_module
from nomad_crystallm.actions.inference import crystallm_inference
from nomad_crystallm.actions.inference.models import (
    CrystallmUserInput,
    InferenceOutput,
    InferenceSettingsInput,
    PromptConstructionInput,
)
from nomad_crystallm.actions.inference.workflow import (
    CrystallmWorkflow,
    InferenceWorkflow,
)


def asset_ref(filename='prompts.csv'):
    return ActionAssetRef(nomad_type='action_asset_ref', filename=filename)


def test_optional_asset_schema_and_round_trip():
    data = CrystallmUserInput(
        user_id='user', upload_id='upload', inference_settings=InferenceSettingsInput()
    )
    assert data.prompts_file is None
    assert data.prompt_construction_inputs == []
    data.prompts_file = asset_ref()
    serialized = data.model_dump_json(by_alias=True)
    assert '"_nomad_type":"action_asset_ref"' in serialized
    assert CrystallmUserInput.model_validate_json(serialized) == data

    schema = CrystallmUserInput.model_json_schema(by_alias=True)
    field = schema['properties']['prompts_file']
    assert field['x-nomad-widget'] == 'file-upload'
    assert '.csv' in field['accept']
    assert '.xlsx' in field['accept']
    assert {'$ref': '#/$defs/ActionAssetRef'} in field['anyOf']
    assert 'prompts_file' not in schema['required']
    assert 'prompt_construction_inputs' not in schema['required']
    assert activity_module.read_prompts in crystallm_inference.load().activities


def test_read_activity_opens_asset_in_owning_workflow(monkeypatch, tmp_path):
    path = tmp_path / 'prompts.csv'
    path.write_text('NaCl,2,P 1\n')
    opener = Mock(side_effect=lambda *_: path.open('rb'))
    monkeypatch.setattr(activity_module, 'open_action_asset', opener)
    environment = ActivityEnvironment()
    ref = asset_ref()

    result = environment.run(activity_module.read_prompts, ref)

    opener.assert_called_once_with(ref, environment.info.workflow_id)
    assert result == [
        PromptConstructionInput(
            composition='NaCl', num_formula_units_per_cell='2', space_group='P 1'
        )
    ]


@pytest.mark.parametrize(
    ('filename', 'contents'),
    [
        ('prompts.xls', b'NaCl'),
        ('prompts.xlsx', b'not a workbook'),
        ('prompts.csv', b'\xff'),
        ('prompts.csv', b'"unterminated'),
    ],
)
def test_unreadable_assets_fail_without_retries(monkeypatch, filename, contents):
    monkeypatch.setattr(
        activity_module, 'open_action_asset', lambda *_: BytesIO(contents)
    )

    with pytest.raises(ApplicationError, match='Unable to read prompts file') as error:
        ActivityEnvironment().run(activity_module.read_prompts, asset_ref(filename))

    assert error.value.non_retryable


def test_missing_asset_fails_without_retries(monkeypatch):
    monkeypatch.setattr(
        activity_module,
        'open_action_asset',
        Mock(side_effect=ValueError('Asset file was not found.')),
    )
    with pytest.raises(ApplicationError, match='was not found') as error:
        ActivityEnvironment().run(activity_module.read_prompts, asset_ref())
    assert error.value.non_retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('manual', 'file_rows'),
    [
        (['NaCl'], None),
        ([], ['NaCl', 'TiO2']),
        (['NaCl'], ['TiO2', 'NaCl']),
        (['NaCl'], []),
        ([], []),
        ([], None),
    ],
)
async def test_combined_prompt_workflow(monkeypatch, manual, file_rows):
    inputs = [PromptConstructionInput(composition=value) for value in manual]
    uploaded = [PromptConstructionInput(composition=value) for value in file_rows or []]
    data = CrystallmUserInput(
        user_id='user',
        upload_id='upload',
        prompts_file=asset_ref() if file_rows is not None else None,
        prompt_construction_inputs=inputs,
        inference_settings=InferenceSettingsInput(num_samples=2, batch_size=3),
    )
    activity_calls, batches, written = [], [], []
    samples = []

    async def execute_activity(function, arg, **_):
        activity_calls.append(function)
        if function is activity_module.read_prompts:
            assert arg == data.prompts_file
            return uploaded
        if function is activity_module.get_prompt:
            return f'prompt:{arg.composition}'
        if function is activity_module.write_results:
            written.append(arg)
        else:
            assert function is activity_module.get_model

    async def execute_child_workflow(function, arg, **_):
        assert function is InferenceWorkflow.run
        batches.append(arg.prompts)
        generated = [
            f'sample-{len(samples) + idx}:{p}' for idx, p in enumerate(arg.prompts)
        ]
        samples.extend(generated)
        return InferenceOutput(generated_samples=generated)

    monkeypatch.setattr(workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(workflow, 'execute_child_workflow', execute_child_workflow)
    monkeypatch.setattr(
        workflow,
        'info',
        lambda: SimpleNamespace(
            workflow_id='test-workflow',
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    combined = manual + (file_rows or [])
    if not combined:
        with pytest.raises(ApplicationError, match='No valid prompts') as error:
            await CrystallmWorkflow().run(data)
        assert error.value.non_retryable
        assert activity_module.get_model not in activity_calls
        assert not batches
        assert not written
        return

    await CrystallmWorkflow().run(data)

    assert [prompt.composition for prompt in data.prompt_construction_inputs] == manual
    expected_prompts = [f'prompt:{value}' for value in combined for _ in range(2)]
    assert [prompt for batch in batches for prompt in batch] == expected_prompts
    assert all(len(batch) <= data.inference_settings.batch_size for batch in batches)
    assert [result.composition for result in written] == combined
    for idx, result in enumerate(written):
        assert result.prompt == f'prompt:{combined[idx]}'
        assert result.generated_samples == samples[idx * 2 : (idx + 1) * 2]
        assert result.relative_cif_dir == f'composition_{idx + 1}_{combined[idx]}'
        assert result.action_instance_id == 'test-workflow'
    if file_rows is None:
        assert activity_module.read_prompts not in activity_calls
    else:
        assert activity_calls[0] is activity_module.read_prompts
