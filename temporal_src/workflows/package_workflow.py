from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_src.activities.package_activity import package_data_activity


@workflow.defn
class PackagingWorkflow:
    @workflow.run
    async def run(self, package_inputs) -> str:
        l0_local_folderpath, atm_cnfg_obj_filepath = package_inputs
        await workflow.execute_activity(
            package_data_activity,
            [l0_local_folderpath, atm_cnfg_obj_filepath],
            start_to_close_timeout=timedelta(seconds=5000),
        )
        return "Processed!!"
