import json
import os
from pathlib import Path
from os.path import dirname
from datetime import datetime
import s3fs
import shutil
import numpy as np
import isofit.utils.template_construction as tmpl
from isofit.utils import surface_model
from types import SimpleNamespace
from utils.iso_atm_utils import extract_data, write_data_to_file, convert_csv_to_envi, l1b_preprocess, download_from_s3, upload_files_to_s3
from utils.apply_oe import apply_oe


def iso_model_init(l0_local_folderpath, l1b_dir):
    # Initialize the S3 filesystem
    fs = s3fs.S3FileSystem(anon=False)

    # Create local tmp image dir
    img_name = l0_local_folderpath.split('/')[-2]
    home_dir = Path(os.path.expanduser("~"))

    # Working directories
    boa_rfl_wdir = os.path.dirname(os.path.dirname(l0_local_folderpath))
    local_wdir = home_dir / img_name
    local_wdir.mkdir(parents=True, exist_ok=True)

    # Check and find metadata path
    def check_path(path, key):
        if path:
            filepath = path[0]
            print(f"{key} file found: {filepath}")
            return f's3://{filepath}'
        else:
            print(f"{key} file not found")
            return None

    # Construct the correct bucket name and key
    bucket_name = boa_rfl_wdir.split('/')[2]
    prefix = '/'.join(boa_rfl_wdir.split('/')[3:])
    metadata = fs.glob(f'{bucket_name}/{prefix}/**/metadata.json')
    l0_meta_filepath = check_path(metadata, "metadata")
    l0_meta_local = local_wdir / "metadata.json"
    download_from_s3(l0_meta_filepath, str(l0_meta_local))

    # metadata
    l0_meta = json.load(open(Path(l0_meta_local)))
    # satellite_id = l0_meta["satellite_id"].split("-")
    # satellite_id = f"{satellite_id[0]}{int(satellite_id[1])}"
    image_id = l0_meta["image_id"]
    to_tif_metadata = fs.glob(f'{boa_rfl_wdir}/to_tif_metadata*.json')
    to_tif_metadata_filepath = check_path(to_tif_metadata, "to_tif_metadata")
    to_tif_metadata_local = local_wdir / "to_tif_metadata.json"
    download_from_s3(to_tif_metadata_filepath, str(to_tif_metadata_local))
    acquisition_date = l0_meta["time_of_capture"]
    iso_data_filepath = local_wdir / "isofit"
    iso_data_filepath.mkdir(parents=True, exist_ok=True)
    print ('iso_data_filepath created:', iso_data_filepath)


    # l1b image paths
    l1b = fs.glob(f'{l1b_dir}*l1b.tif')
    l1b_filepath = check_path(l1b, "l1b")
    l1b_filepath_local = iso_data_filepath / "l1b.tif"
    download_from_s3(l1b_filepath, str(l1b_filepath_local))

    loc = fs.glob(f'{l1b_dir}*l1b_geodetic_mask.tif')
    loc_filepath = check_path(loc, "loc")
    loc_filepath_local = iso_data_filepath / "l1b_geodetic_mask.tif"
    download_from_s3(loc_filepath, loc_filepath_local)

    sol = fs.glob(f'{l1b_dir}*l1b_solar_mask.tif')
    sol_filepath = check_path(sol, "sol")
    sol_filepath_local = iso_data_filepath / "l1b_solar_mask.tif"
    download_from_s3(sol_filepath, sol_filepath_local)

    obs = fs.glob(f'{l1b_dir}*l1b_view_vector_mask.tif')
    obs_filepath = check_path(obs, "obs")
    obs_filepath_local = iso_data_filepath / "l1b_view_vector_mask.tif"
    download_from_s3(obs_filepath, obs_filepath_local)
    

    # input data paths
    input_rad = iso_data_filepath / f"{image_id}_rdn"
    input_loc = iso_data_filepath / f"{image_id}_loc"
    input_obs = iso_data_filepath / f"{image_id}_obs"
    waves_filepath = iso_data_filepath / "wavelengths.txt"
    surface_filepath = iso_data_filepath / "surface.mat"

    # emulator path
    emulator_base = "/home/ray/sRTMnet_v100/sRTMnet_v100.h5"

    # surface paths 
    bucket_name = "d-ipr-services-s3-01"
    aws_region = "us-east-2"
    aster_filepath = "s3://d-ipr-services-s3-01/satellite_image_processing/isofit_data/jpl_aster.csv"
    ucsb_filepath = "s3://d-ipr-services-s3-01/satellite_image_processing/isofit_data/ucsb_urban.csv"
    aquatic_filepath = "s3://d-ipr-services-s3-01/satellite_image_processing/isofit_data/aquatic_rrs_1nm.csv"
    aster_local_filepath = str(iso_data_filepath) + "/jpl_aster.csv"
    ucsb_local_filepath = str(iso_data_filepath) + "/ucsb_urban.csv"
    aquatic_local_filepath = str(iso_data_filepath) + "/aquatic_rrs.csv"
    aster_envi_filepath = str(iso_data_filepath) + "/jpl_aster"
    ucsb_envi_filepath = str(iso_data_filepath) + "/ucsb_urban"
    aquatic_envi_filepath = str(iso_data_filepath) + "/aquatic_rrs"
    surface_tmp_filepath = "s3://d-ipr-services-s3-01/satellite_image_processing/isofit_data/surface_temp.json"
    surface_tmp_local_filepath = str(iso_data_filepath) + "/surface_temp.json"
    surface_edit_filepath = str(iso_data_filepath) + "/surface_edit.json"
    

    # isofit settings
    # Parse the date string into a datetime object
    date_object = datetime.strptime(acquisition_date, '%Y-%m-%dT%H:%M:%SZ')
    # Convert the datetime object to the desired string format
    date = date_object.strftime('%Y%m%d') # YYYYMMDD
    log_file = iso_data_filepath / "isofit.log"
    sensor = f'NA-{date}' 

    atm_cnfg_init = SimpleNamespace(
        working_directory=str(iso_data_filepath),
        to_tif_meta_filepath=to_tif_metadata_filepath,
        to_tif_meta_local_filepath = str(to_tif_metadata_local),
        l0_meta_filepath=l0_meta_filepath,
        l0_meta_local_filepath=str(l0_meta_local),
        iso_data_filepath=str(iso_data_filepath),
        atm_outfolderpath=boa_rfl_wdir + "/isofit/",
        image_id=image_id,
        acquisition_date=acquisition_date,
        bucket_name=bucket_name,
        aws_region=aws_region,
        l1b_filepath=l1b_filepath,
        l1b_filepath_local=str(l1b_filepath_local),
        loc_filepath=loc_filepath,
        loc_filepath_local=str(loc_filepath_local),
        sol_filepath=sol_filepath,
        sol_filepath_local=str(sol_filepath_local),
        obs_filepath=obs_filepath,
        obs_filepath_local=str(obs_filepath_local),
        input_radiance=str(input_rad),
        input_loc=str(input_loc),
        input_obs=str(input_obs),
        wavelength_path=str(waves_filepath),
        surface_path=str(surface_filepath),
        emulator_base=emulator_base,
        aster_path=aster_filepath,
        aster_local_path=aster_local_filepath,
        ucsb_path=ucsb_filepath,
        ucsb_local_path=ucsb_local_filepath,
        aquatic_path=aquatic_filepath,
        aquatic_local_path=aquatic_local_filepath,
        surface_temp_path=surface_tmp_filepath,
        surface_temp_local_path=surface_tmp_local_filepath,
        surface_edit_path=surface_edit_filepath,
        aster_envi_path=aster_envi_filepath,
        ucsb_envi_path=ucsb_envi_filepath,
        aquatic_envi_path=aquatic_envi_filepath,
        log_file=str(log_file),
        sensor=sensor,
        uncorrelated_radiometric_uncertainty=0.1,
        eps=1e-6,
        n_cores=10, # double check with karthik here
        segmentation_size=400,
        num_neighbors=[20],
        inversion_windows=[[380,920]],
        chunksize=256,
        empirical_line=True,
        analytical_line=False,
        atmosphere_type="ATM_MIDLAT_SUMMER",
        surface_category="multicomponent_surface",
        n_pca=3,
        copy_input_files=False,
        presolve=True,
        debug_args=False,
        logging_level="INFO",
        multiple_restarts=False,
        atm_sigma=[2],
        prebuilt_lut=None,
        model_discrepancy_path=None,
        aerosol_climatology_path=None,
        channelized_uncertainty_path=None,
        rdn_factors_path=None,
        modtran_path=None,
        pressure_elevation=False,
        lut_config_path=None,
        ray_temp_dir="/tmp/ray",
        rt_cleanup_list=["*r_k", "*t_k", "*tp7", "*wrn", "*psc", "*plt", "*7sc", "*acd"],
    )

    atm_cnfg_init = vars(atm_cnfg_init)
    atm_cnfg_init_json = json.dumps(atm_cnfg_init)

    return atm_cnfg_init_json



def preprocess(atm_cnfg):

    # Convert JSON string to SimpleNamespace
    atm_cnfg = json.loads(atm_cnfg, object_hook=lambda d: SimpleNamespace(**d))

    # build wavelength file
    wavelengths, fwhms = extract_data(atm_cnfg.to_tif_meta_local_filepath)
    write_data_to_file(atm_cnfg.wavelength_path, wavelengths, fwhms)

    # convert surface csvs to envi
    # convert surface aster csv
    convert_csv_to_envi(atm_cnfg.aster_path, atm_cnfg.aster_local_path)

    # convert surface ucsb csv
    convert_csv_to_envi(atm_cnfg.ucsb_path, atm_cnfg.ucsb_local_path)
    
    # convert surface aquatic csv
    convert_csv_to_envi(atm_cnfg.aquatic_path, atm_cnfg.aquatic_local_path)

    # l1b preprocessing
    l1b_preprocess(atm_cnfg)

    atm_cnfg = vars(atm_cnfg)
    atm_cnfg_json = json.dumps(atm_cnfg)

    return atm_cnfg_json


def build_surface(atm_cnfg):

    # Convert JSON string to SimpleNamespace
    atm_cnfg = json.loads(atm_cnfg, object_hook=lambda d: SimpleNamespace(**d))

    # download surface model template
    download_from_s3(atm_cnfg.surface_temp_path, atm_cnfg.surface_temp_local_path)

    # set paths
    output_surface_model_file = atm_cnfg.surface_path
    wl_out_file_path = atm_cnfg.wavelength_path
    aster_img_path = atm_cnfg.aster_envi_path
    ucsb_img_path = atm_cnfg.ucsb_envi_path
    aquatic_img_path = atm_cnfg.aquatic_envi_path
    surface_json_path = atm_cnfg.surface_temp_local_path
    out_json_path = atm_cnfg.surface_edit_path

    if not os.path.isfile(output_surface_model_file):
        print("Surface model does not exist. Creating surface model...")

        print("Building surface json config file")
        with open(surface_json_path, 'r') as file:
            filedata = file.read()

        filedata = filedata.replace('${output_model_file}', output_surface_model_file)
        filedata = filedata.replace('${wavelength_file}', wl_out_file_path)
        filedata = filedata.replace('${input_spectrum_aster}', aster_img_path)
        filedata = filedata.replace('${input_spectrum_ucsb}', ucsb_img_path)
        filedata = filedata.replace('${input_spectrum_aquatic}', aquatic_img_path)

        with open(out_json_path, 'w') as file:
            file.write(filedata)

        # build surface model
        print("Building surface model using config file")
        surface_model(out_json_path)
    else:
        print("Surface model found. Continuing...")
    
    # create and stage paths/directories for isofit
    print ('Staging files for isofit...')
    iso_cnfg = tmpl.Pathnames(atm_cnfg)
    iso_cnfg.make_directories()
    iso_cnfg.stage_files()

    # combine iso and phsicor atm configs into one
    iso_cnfg_dict = vars(iso_cnfg)
    atm_cnfg2 = {**iso_cnfg_dict, **vars(atm_cnfg)}
    atm_cnfg2 = json.dumps(atm_cnfg2)
    print ('Files staged.')

    return atm_cnfg2


def run_apply_oe(atm_cnfg):
    # run optimal estimation method
    print ('Running optimal estimation method...')

    # Convert JSON string to SimpleNamespace
    atm_cnfg = json.loads(atm_cnfg, object_hook=lambda d: SimpleNamespace(**d))
    
    # apply OE
    apply_oe(atm_cnfg)

    print ('Atmospheric Correction Completed Successfully!')

def cleanup (atm_cnfg):
    # Convert JSON string to SimpleNamespace
    atm_cnfg = json.loads(atm_cnfg, object_hook=lambda d: SimpleNamespace(**d))

    upload_list = [f'{atm_cnfg.image_id}_rfl_rfl', f'{atm_cnfg.image_id}_uncert_uncert', 
                   f'{atm_cnfg.image_id}_rfl_rfl.hdr', f'{atm_cnfg.image_id}_uncert_uncert.hdr', 
                   'isofit.log', f'{atm_cnfg.image_id}_rdn_isofit.json', 'wavelengths.txt']

    # Upload the output files to S3
    upload_files_to_s3(atm_cnfg.atm_outfolderpath, str(atm_cnfg.iso_data_filepath), upload_list)

    # Delete the local directory
    shutil.rmtree(atm_cnfg.iso_data_filepath)
    print(f"Deleted local directory: {atm_cnfg.iso_data_filepath}")

    print ('Pixxel Isofit Routine Finished.')
