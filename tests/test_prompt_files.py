import csv
from io import BytesIO, StringIO
from unittest.mock import Mock

import pytest
from openpyxl import Workbook

from nomad_crystallm.actions.inference.prompt_files import read_prompt_inputs

HEADER = ['composition', 'num_formula_units_per_cell', 'space_group']


def table_file(rows, suffix):
    if suffix == '.csv':
        text = StringIO(newline='')
        csv.writer(text).writerows(rows)
        return BytesIO(text.getvalue().encode('utf-8-sig'))
    workbook = Workbook()
    for row in rows:
        workbook.active.append(row)
    # Additional sheets are not prompt sources, even when active.
    workbook.create_sheet('Ignored').append(['Fe', 8, 'P 1'])
    workbook.active = 1
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize('suffix', ['.csv', '.xlsx'])
@pytest.mark.parametrize('header', [True, False])
def test_read_prompt_table(suffix, header):
    rows = [[], HEADER] if header else [[]]
    rows += [
        [' NaCl ', 2.0, ' P 1 '],
        ['TiO2'],
        ['NaN', '', ''],
        [],
        ['NaCl', 2, 'P 1'],
    ]
    logger = Mock()

    result = read_prompt_inputs(table_file(rows, suffix), f'prompts{suffix}', logger)

    assert [prompt.model_dump() for prompt in result] == [
        {
            'composition': 'NaCl',
            'num_formula_units_per_cell': '2',
            'space_group': 'P 1',
        },
        {
            'composition': 'TiO2',
            'num_formula_units_per_cell': '1',
            'space_group': '',
        },
        {
            'composition': 'NaN',
            'num_formula_units_per_cell': '1',
            'space_group': '',
        },
        {
            'composition': 'NaCl',
            'num_formula_units_per_cell': '2',
            'space_group': 'P 1',
        },
    ]
    logger.warning.assert_not_called()


@pytest.mark.parametrize('suffix', ['.csv', '.xlsx'])
def test_invalid_rows_are_skipped_with_row_numbers(suffix):
    logger = Mock()
    rows = [
        HEADER,
        ['', 1, ''],
        ['NotAnElement', 1, ''],
        ['NaCl', 5, ''],
        ['NaCl', 2.5, ''],
        ['NaCl', '2.0000000000000000001', ''],
        ['NaCl', 1, 'invalid group'],
        ['NaCl', 1, '', 'unexpected'],
        ['NaCl', 1, ''],
    ]

    result = read_prompt_inputs(table_file(rows, suffix), f'prompts{suffix}', logger)

    assert [prompt.composition for prompt in result] == ['NaCl']
    assert [call.args[1] for call in logger.warning.call_args_list] == list(range(2, 9))
    assert all(
        call.args[2] == f'prompts{suffix}' for call in logger.warning.call_args_list
    )


@pytest.mark.parametrize('suffix', ['.csv', '.xlsx'])
@pytest.mark.parametrize('rows', [[], [HEADER], [['NotAnElement']]])
def test_no_valid_rows_returns_empty_list(suffix, rows):
    assert (
        read_prompt_inputs(table_file(rows, suffix), f'prompts{suffix}', Mock()) == []
    )


def test_unsupported_file_type():
    with pytest.raises(ValueError, match='.csv or .xlsx'):
        read_prompt_inputs(BytesIO(b'NaCl'), 'prompts.xls', Mock())
