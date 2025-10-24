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
from nomad import infrastructure
from nomad.processing.data import Entry, Upload


def get_upload(upload_id: str, user_id: str) -> Upload | None:
    """
    Retrieve an upload by ID and check if the user is authorized to access it.
    Args:
        upload_id (str): The ID of the upload to retrieve.
        user_id (str): The ID of the user requesting access.
    Returns:
        Upload | None: The upload object if found and authorized, else None.
    """
    if infrastructure.mongo_client is None:
        infrastructure.setup_mongo()

    upload = Upload.get(upload_id)

    if upload is None:
        return None

    # Determine if user is authorized to get the upload.
    is_coauthor = isinstance(upload.coauthors, list) and user_id in upload.coauthors
    is_authorized = upload.main_author == user_id or is_coauthor

    # Raise error if not authorized
    if not is_authorized:
        raise PermissionError(
            f'User {user_id} is not authorized to access upload {upload_id}.'
        )

    return upload


def get_reference_from_mainfile(
    upload_id: str, mainfile: str, archive_path: str = 'data'
) -> str:
    """
    Uses the upload_id and mainfile to find the entry_id of an entry and returns
    a MProxy reference of a section in the entry.

    Args:
        upload_id (str): Upload ID of the upload in which the entry is located.
        mainfile (str): Mainfile of the entry to be referenced.
        archive_path (str, Optional): Path in the entry where the section is located.
            Defaults to 'data'.

    Returns:
        str: _description_
    """
    entry_id = None
    for entry in Entry.objects(upload_id=upload_id):
        if entry.mainfile == mainfile:
            entry_id = entry.entry_id
    if entry_id is None:
        return None
    return f'../uploads/{upload_id}/archive/{entry_id}#/{archive_path}'
