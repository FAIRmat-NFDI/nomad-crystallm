#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest


# TODO: activate the test only temporal-workflows branch is merged in nomad-FAIR
@pytest.mark.skip(
    'This test is skipped until the temporal-workflows branch is merged in nomad-FAIR'
)
def test_inference_result(caplog):
    """
    Test the processing of system based on the CIF file.
    """
    from nomad.client import normalize_all, parse

    archive = parse('tests/data/inference_result.archive.yaml')[0]
    normalize_all(archive)

    assert archive.data.generated_structures[0].chemical_formula_iupac == 'NaCl'
    assert archive.results.material.topology[0].chemical_formula_iupac == 'NaCl'
