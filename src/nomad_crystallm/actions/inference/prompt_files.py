"""Read prompt tables without loading the inference model or its dependencies."""

import csv
from collections.abc import Iterator, Sequence
from decimal import Decimal, InvalidOperation
from io import TextIOWrapper
from logging import Logger, LoggerAdapter
from pathlib import Path
from typing import BinaryIO, get_args

from openpyxl import load_workbook
from pymatgen.core import Composition

from nomad_crystallm.actions.inference.models import (
    NumFormulaUnitsPerCellLiteral,
    PromptConstructionInput,
)

PROMPT_COLUMNS = ('composition', 'num_formula_units_per_cell', 'space_group')
FORMULA_UNITS = {
    Decimal(value): value for value in get_args(NumFormulaUnitsPerCellLiteral)
}


def _iter_rows(file: BinaryIO, filename: str) -> Iterator[Sequence[object]]:
    suffix = Path(filename).suffix.lower()
    if suffix == '.csv':
        with TextIOWrapper(file, encoding='utf-8-sig', newline='') as text:
            yield from csv.reader(text, strict=True)
    elif suffix == '.xlsx':
        workbook = load_workbook(file, read_only=True, data_only=True)
        try:
            yield from workbook.worksheets[0].iter_rows(values_only=True)
        finally:
            workbook.close()
    else:
        raise ValueError('Prompts files must use the .csv or .xlsx format.')


def _normalize_formula_units(value: str) -> str:
    if not value:
        return '1'
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return FORMULA_UNITS.get(number, value) if number.is_finite() else value


def read_prompt_inputs(
    file: BinaryIO,
    filename: str,
    logger: Logger | LoggerAdapter,
) -> list[PromptConstructionInput]:
    """Return valid rows in file order, logging rejected rows with their positions."""
    prompts = []
    first_row = True
    for row_number, row in enumerate(_iter_rows(file, filename), start=1):
        values = ['' if value is None else str(value).strip() for value in row]
        # Spreadsheet readers include empty trailing cells in otherwise short rows.
        while values and not values[-1]:
            values.pop()
        if not values:
            continue
        if first_row:
            first_row = False
            if tuple(values) == PROMPT_COLUMNS:
                continue
        try:
            if len(values) > len(PROMPT_COLUMNS):
                raise ValueError('Expected at most three populated columns.')
            values.extend([''] * (len(PROMPT_COLUMNS) - len(values)))
            if not values[0]:
                raise ValueError('Composition is required.')
            prompt = PromptConstructionInput(
                composition=values[0],
                num_formula_units_per_cell=_normalize_formula_units(values[1]),
                space_group=values[2],
            )
            Composition(prompt.composition, strict=True)
        except ValueError as error:
            logger.warning(
                'Skipping invalid prompt row %s in %s: %s',
                row_number,
                filename,
                error,
            )
            continue
        prompts.append(prompt)
    return prompts
