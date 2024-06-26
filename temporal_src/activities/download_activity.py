from temporalio import activity

from phsicor.download.download_data import download_data
from temporal_src.config import config


@activity.defn
async def download_activity(l0_package_url: str) -> str:
    """Temporal activity for l0 data package download module"""
    local_l0_folderpath = download_data(l0_package_url, config.ROOT_FOLDER)
    return local_l0_folderpath
