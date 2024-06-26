from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_src.activities.geometric_activity import (
        dem_activity,
        geoproject_activity,
        quaternion_activity,
        truth_activity,
    )


@workflow.defn
class GeometricWorkflow:
    @workflow.run
    async def run(self, l0_local_folderpath: str) -> str:
        dem_out, geo_cnfg = await workflow.execute_activity(
            dem_activity,
            l0_local_folderpath,
            start_to_close_timeout=timedelta(seconds=5),
        )

        truth_out = await workflow.execute_activity(
            truth_activity,
            [dem_out, geo_cnfg],
            start_to_close_timeout=timedelta(seconds=500),
        )
        await workflow.execute_activity(
            quaternion_activity,
            [dem_out, geo_cnfg, truth_out],
            start_to_close_timeout=timedelta(seconds=500),
        )

        output_dir = await workflow.execute_activity(
            geoproject_activity,
            [dem_out, geo_cnfg],
            start_to_close_timeout=timedelta(seconds=500),
        )
        return output_dir
