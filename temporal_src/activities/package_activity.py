import warnings

from temporalio import activity

warnings.filterwarnings("ignore")

from phsicor.package.packaging_main import package_data


@activity.defn
async def package_data_activity(package_inputs):
    l0_local_folderpath, atm_cnfg_obj_filepath = package_inputs
    package_data(l0_local_folderpath, atm_cnfg_obj_filepath)
