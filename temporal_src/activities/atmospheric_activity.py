import warnings

from temporalio import activity

warnings.filterwarnings("ignore")

from phsicor.atmospheric.atm_main import (
    apply_preclassification,
    atm_model_init,
    generate_corrected_data,
    generate_l1b_l1c,
    generate_visibility_map,
    generate_water_vapor_map,
)


@activity.defn
async def atm_model_init_activity(atm_inputs):
    l0_local_folderpath, l1b_dir = atm_inputs
    atm_cnfg = atm_model_init(l0_local_folderpath, l1b_dir)
    return atm_cnfg


@activity.defn
async def generate_l1b_l1c_activity(atm_cnfg):
    luts_path = generate_l1b_l1c(atm_cnfg)
    return luts_path


@activity.defn
async def preclassification_activity(atm_cnfg):
    apply_preclassification(atm_cnfg)


@activity.defn
async def visibility_map_activity(vis_inputs):
    atm_cnfg, lut_paths = vis_inputs
    atm_cnfg, aerosol_type = generate_visibility_map(atm_cnfg, lut_paths)
    return atm_cnfg, aerosol_type


@activity.defn
async def water_vapor_map_activity(wv_inputs):
    atm_cnfg, lut_paths, aerosol_type = wv_inputs
    generate_water_vapor_map(atm_cnfg, lut_paths, aerosol_type)


@activity.defn
async def generate_l2a_activity(l2a_inputs):
    atm_cnfg, lut_paths, aerosol_type = l2a_inputs
    generate_corrected_data(atm_cnfg, lut_paths, aerosol_type)
