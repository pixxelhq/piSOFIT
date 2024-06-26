from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_src.activities.download_activity import download_activity


@workflow.defn
class DownloadWorkflow:
    @workflow.run
    async def run(self, l0_package_url: str) -> str:
        result = await workflow.execute_activity(
            download_activity,
            l0_package_url,
            start_to_close_timeout=timedelta(seconds=5),
        )
        return result
