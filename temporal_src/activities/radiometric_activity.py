from typing import List

import numpy as np
from temporalio import activity

from phsicor.radiometric.radiometric_main import (
    perform_denoise,
    perform_flat_field_corr,
    perform_pattern_noise_corr,
)


@activity.defn
async def flat_field_corr_activity(l0_local_folderpath: str) -> np.ndarray:
    """Temporal activity for flat field correction"""
    sat_id, nuc_corr_folderpath, image_id = perform_flat_field_corr(l0_local_folderpath)
    return sat_id, nuc_corr_folderpath, image_id


@activity.defn
async def pattern_noise_corr_activity(destrip_inputs: List[str]) -> str:
    """Temporal activity for pattern noise correction"""
    sat_id, nuc_corr_folderpath = destrip_inputs
    destrip_folderpath = perform_pattern_noise_corr(sat_id, nuc_corr_folderpath)
    return destrip_folderpath


@activity.defn
async def denoise_activity(denoise_inputs) -> str:
    """Temporal activity for denoise module"""
    destrip_folderpath, satellite_id, image_id = denoise_inputs
    denoise_folderpath = perform_denoise(destrip_folderpath, satellite_id, image_id)
    return denoise_folderpath
