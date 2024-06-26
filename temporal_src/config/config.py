from pathlib import Path

from decouple import config

TEMPORAL_SERVER_URL = config("TEMPORAL_SERVER_URL", default="localhost:7233")
TEMPORAL_NAMESPACE = config("TEMPORAL_NAMESPACE", default="default")
TASK_QUEUE = config("CORRECTION_QUEUE", default="l0-l2a-correction-tasks")

ROOT_FOLDER = config("DATA_FOLDER", default="/phsicor3/data/")
CODE_FOLDER = config("CODE_FOLDER", default="/phsicor3/phsicor")

nuc_fileinfo = {}
nuc_fileinfo["FF00"] = {}
nuc_fileinfo["FF00"][
    "VNIR01"
] = "/Users/phsicor3/data/d-ipr-td1-s3-01/VNIR/LUTs/TD1_NUC_B179.npy"
nuc_fileinfo["FF00"][
    "VNIR02"
] = "/Users/phsicor3/data/d-ipr-td1-s3-01/VNIR/LUTs/TD1_NUC_B179.npy"
nuc_fileinfo["FF00"][
    "VNIR03"
] = "/Users/phsicor3/data/d-ipr-td1-s3-01/VNIR/LUTs/TD1_NUC_B179.npy"
nuc_fileinfo["FF00"][
    "VNIR04"
] = "/Users/phsicor3/data/d-ipr-td1-s3-01/VNIR/LUTs/TD1_NUC_B179.npy"
nuc_fileinfo["TD1"] = {}
nuc_fileinfo["TD1"][
    "VNIR01"
] = "/Users/phsicor3/data/d-ipr-td1-s3-01/VNIR/LUTs/TD1_NUC_B179.npy"
nuc_fileinfo["TD2"] = {}
nuc_fileinfo["TD2"]["VNIR01"] = {}
nuc_fileinfo["TD2"]["VNIR01"][
    "20221231"
] = "/Users/phsicor3/data/d-ipr-td2-s3-01/VNIR/LUTs/TD2_NUC_V1_610.npy"
nuc_fileinfo["TD2"]["VNIR01"][
    "20230430"
] = "/Users/phsicor3/data/d-ipr-td2-s3-01/VNIR/LUTs/TD2_NUC_V3_660.npy"
nuc_fileinfo["TD2"]["VNIR01"][
    "20230501"
] = "/Users/phsicor3/data/d-ipr-td2-s3-01/VNIR/LUTs/TD2_NUC_V4_22092023_4cal_images.npy"
nuc_fileinfo["TD2"]["VNIR01"][
    "20230901"
] = "/Users/phsicor3/data/d-ipr-td2-s3-01/VNIR/LUTs/TD2_NUC_V5_13102023.npy"
nuc_fileinfo["TD2"]["VNIR01"][
    "20231001"
] = "/Users/phsicor3/data/d-ipr-td2-s3-01/VNIR/LUTs/TD2_NUC_V5_OCT_08112023.npy"

S3_LUTS_PATH = config(
    "LUTS_URL",
    default="s3://d-ipr-services-s3-01/phsicor3/satdata/d-ipr-td2-s3-01/VNIR/LUTs/",
)

GEO_LUTS = config(
    "GEOMODEL_LUTs",
    default=[
        "ingredient_credentials.json",
        "TD1_camera_3_default_bandset_1.json",
        "TD2_camera_3.json",
        "TD2_camera_4.json",
        "us_nga_egm2008_1.tif",
    ],
)

ATM_LUTS = config(
    "ATMMODEL_LUTs", default=["TD_Band_Selection_Worksheet.xlsx", "lut.h5"]
)

aerosol = config("AEROSOL", default="rural")

# iso data paths that need storage on s3 #
# 1. surface tmp json file: surface_tmp_filepath.json
# 2. surface edited json file: surface_edit_filepath.json
# 3. aster speclib csv file: jpl_aster.csv
# 4. ucsb speclib csv file: ucsb_urban.csv
# 5. aquatic speclib csv file: aquatic_rrs.csv
# 6. 6SV1 RTM and path added to system environment variables (bash_profile)
# 7. sRTMnet modtran emulator and path added to system environment variables (bash_profile)
