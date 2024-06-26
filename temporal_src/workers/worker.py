import asyncio
import sys
from pathlib import Path

code_source_path = str(Path(__file__).parent.resolve()).split("temporal_src")[0]
sys.path.append(code_source_path)
# sys.path.append(Path(code_source_path)/"phsicor/geometric/L1B_processor/src/adjust/")

from temporalio.client import Client
from temporalio.worker import Worker

from temporal_src.activities.atmospheric_activity import (
    atm_model_init_activity,
    generate_l1b_l1c_activity,
    generate_l2a_activity,
    preclassification_activity,
    visibility_map_activity,
    water_vapor_map_activity,
)
from temporal_src.activities.download_activity import download_activity
from temporal_src.activities.geometric_activity import (
    dem_activity,
    geoproject_activity,
    quaternion_activity,
    truth_activity,
)
from temporal_src.activities.package_activity import package_data_activity
from temporal_src.activities.radiometric_activity import (
    denoise_activity,
    flat_field_corr_activity,
    pattern_noise_corr_activity,
)
from temporal_src.config import config
from temporal_src.workflows.atmospheric_workflow import AtmosphericWorkflow
from temporal_src.workflows.download_workflow import DownloadWorkflow
from temporal_src.workflows.geometric_workflow import GeometricWorkflow
from temporal_src.workflows.package_workflow import PackagingWorkflow
from temporal_src.workflows.radiometric_workflow import RadiometricWorkflow


async def main():
    """Temporal worker"""
    client = await Client.connect(
        config.TEMPORAL_SERVER_URL, namespace=config.TEMPORAL_NAMESPACE
    )
    # Run the worker
    worker = Worker(
        client,
        task_queue="l0-l2a-correction-tasks",
        workflows=[
            DownloadWorkflow,
            RadiometricWorkflow,
            GeometricWorkflow,
            AtmosphericWorkflow,
            PackagingWorkflow,
        ],
        activities=[
            download_activity,
            flat_field_corr_activity,
            pattern_noise_corr_activity,
            denoise_activity,
            dem_activity,
            truth_activity,
            quaternion_activity,
            geoproject_activity,
            atm_model_init_activity,
            generate_l1b_l1c_activity,
            preclassification_activity,
            visibility_map_activity,
            water_vapor_map_activity,
            generate_l2a_activity,
            package_data_activity,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
