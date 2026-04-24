from temporalio import activity

from nomad.actions.assets import resolve_action_asset_path
from nomad.actions.assets.models import ActionAssetRef


@activity.defn
def generate_random_number_activity(lower_bound: int, upper_bound: int) -> int:
    """
    Generate a random integer between lower_bound and upper_bound (inclusive).
    """
    import random

    if lower_bound > upper_bound:
        raise ValueError('lower_bound must be less than or equal to upper_bound')

    return random.randint(lower_bound, upper_bound)


def _asset_metadata(
    action_instance_id: str, asset_ref: ActionAssetRef, label: str
) -> dict[str, str | int | None]:
    path = resolve_action_asset_path(asset_ref, action_instance_id)
    return {
        'label': label,
        'filename': asset_ref.filename,
        'resolved_path': str(path),
        'size_bytes': path.stat().st_size,
        'media_type': asset_ref.media_type,
    }


@activity.defn
def collect_signal_asset_metadata_activity(
    action_instance_id: str,
    image_path: ActionAssetRef,
    audio_submission: ActionAssetRef,
) -> dict[str, dict[str, str | int | None]]:
    """
    Resolve user-submitted assets and return minimal metadata for each file.
    """
    return {
        'image_path': _asset_metadata(action_instance_id, image_path, 'image_path'),
        'audio_submission': _asset_metadata(
            action_instance_id, audio_submission, 'audio_submission'
        ),
    }
