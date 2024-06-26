from typing import Dict, List

import numpy as np
from temporalio import activity

from phsicor.geometric.geo_main import (
    dem,
    geoproject,
    get_geo_model_config,
    quaternion,
    truth,
)


@activity.defn
async def dem_activity(l0_local_folderpath: str) -> np.ndarray:
    """Temporal activity for DEM data download."""
    geo_cnfg = get_geo_model_config(l0_local_folderpath)
    dem_out = dem(
        geo_cnfg["oat_path"],
        geo_cnfg["l0_metadata_path"],
        geo_cnfg["L1A_dir"],
        geo_cnfg["camera_dir"],
        geo_cnfg["satellite_id"],
    )

    return dem_out, geo_cnfg


@activity.defn
async def truth_activity(truth_inputs) -> Dict:
    """Temporal activity for truth reference data download."""
    dem_out, geo_cnfg = truth_inputs
    truth_out = truth(geo_cnfg["camera_dir"], geo_cnfg["L1A_dir"], dem_out)
    return truth_out


@activity.defn
async def quaternion_activity(qa_inputs) -> str:
    """Temporal activity for quarternion adjuster"""
    dem_out, geo_cnfg, truth_out = qa_inputs
    quaternion(dem_out, geo_cnfg["L1A_dir"], truth_out)


@activity.defn
async def geoproject_activity(warping_inputs) -> str:
    """Temporal activity for band warping"""
    dem_out, geo_cnfg = warping_inputs
    output_dir = geoproject(dem_out, geo_cnfg["L1A_dir"])
    return output_dir
