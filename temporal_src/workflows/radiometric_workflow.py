from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_src.activities.radiometric_activity import (
        denoise_activity,
        flat_field_corr_activity,
        pattern_noise_corr_activity,
    )


@workflow.defn
class RadiometricWorkflow:
    @workflow.run
    async def run(self, l0_local_folderpath: str) -> str:
        sat_id, nuc_corr_folderpath, image_id = await workflow.execute_activity(
            flat_field_corr_activity,
            l0_local_folderpath,
            start_to_close_timeout=timedelta(seconds=5),
        )

        destrip_folderpath = await workflow.execute_activity(
            pattern_noise_corr_activity,
            [sat_id, nuc_corr_folderpath],
            start_to_close_timeout=timedelta(seconds=5),
        )

        denoise_folderpath = await workflow.execute_activity(
            denoise_activity,
            [destrip_folderpath, sat_id, image_id],
            start_to_close_timeout=timedelta(seconds=5),
        )
        return denoise_folderpath
