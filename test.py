import argparse
from isofit_main import iso_model_init, preprocess, build_surface, run_apply_oe, cleanup

def process_images(l0_local_folderpath, l1b_dir):
    atm_cnfg = iso_model_init(l0_local_folderpath, l1b_dir)
    atm_cnfg = preprocess(atm_cnfg)
    atm_cnfg_iso = build_surface(atm_cnfg)
    run_apply_oe(atm_cnfg_iso)
    cleanup(atm_cnfg_iso)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process satellite images using piSOFIT.')
    parser.add_argument('l0_local_folderpath', type=str, help='Path to the L0 s3 folder.')
    parser.add_argument('l1b_dir', type=str, help='Path to the s3 L1B directory.')

    args = parser.parse_args()
    process_images(args.l0_local_folderpath, args.l1b_dir)


# #%%

# l0_local_folderpath = 's3://d-ipr-services-s3-01/satellite_image_processing/products/TD1/image_007480/image_007480_l0/'
# l1b_dir = 's3://d-ipr-services-s3-01/satellite_image_processing/geometric_modelling__experiments/Pixxel-TD1-L1B/7480/'

# #%
# from isofit_main import iso_model_init
# atm_cnfg = iso_model_init(l0_local_folderpath, l1b_dir)

# #%
# from isofit_main import preprocess
# atm_cnfg = preprocess(atm_cnfg)

# #%
# from isofit_main import build_surface
# atm_cnfg_iso = build_surface(atm_cnfg)

# #%
# from isofit_main import run_apply_oe
# run_apply_oe(atm_cnfg_iso)

# #%
# from isofit_main import cleanup
# cleanup(atm_cnfg_iso)