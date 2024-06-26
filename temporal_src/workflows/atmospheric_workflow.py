from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_src.activities.atmospheric_activity import (
        atm_model_init_activity,
        generate_l1b_l1c_activity,
        generate_l2a_activity,
        preclassification_activity,
        visibility_map_activity,
        water_vapor_map_activity,
    )


@workflow.defn
class AtmosphericWorkflow:
    @workflow.run
    async def run(self, atm_input) -> str:
        l0_local_folderpath, l1b_dir = atm_input
        atm_cnfg = await workflow.execute_activity(
            atm_model_init_activity,
            [l0_local_folderpath, l1b_dir],
            start_to_close_timeout=timedelta(seconds=3000),
        )
        lut_paths = await workflow.execute_activity(
            generate_l1b_l1c_activity,
            atm_cnfg,
            start_to_close_timeout=timedelta(seconds=3000),
        )

        await workflow.execute_activity(
            preclassification_activity,
            atm_cnfg,
            start_to_close_timeout=timedelta(seconds=3000),
        )

        atm_cnfg, aerosol_type = await workflow.execute_activity(
            visibility_map_activity,
            [atm_cnfg, lut_paths],
            start_to_close_timeout=timedelta(seconds=3000),
        )

        await workflow.execute_activity(
            water_vapor_map_activity,
            [atm_cnfg, lut_paths, aerosol_type],
            start_to_close_timeout=timedelta(seconds=3000),
        )

        await workflow.execute_activity(
            generate_l2a_activity,
            [atm_cnfg, lut_paths, aerosol_type],
            start_to_close_timeout=timedelta(seconds=5000),
        )

        return atm_cnfg["atm_cnfg_obj_filepath"]
