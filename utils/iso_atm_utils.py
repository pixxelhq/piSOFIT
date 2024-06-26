import json
import pandas as pd
import spectral.io.envi as envi
import boto3
import s3fs
import datetime as dt
import os
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import hytools as ht
from hytools.io.envi import WriteENVI,envi_header_dict
from hytools.topo.topo import calc_cosine_i
import numpy as np
import pandas as pd
from pyproj import CRS
import rioxarray
import xarray as xr
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from boto3.s3.transfer import TransferConfig


def download_from_s3(s3_url: str, destination: str) -> None:
    """Download a file or folder from an S3 bucket to local.
    Skips download if exists in local.

    :param s3_url: The URL of a folder or a file on s3 to download.
    :type s3_url: str
    :param destination: The local directory to save the folder.
    :type destination: str
    """
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip("/")

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=object_key)
    list_objects = response.get("Contents", [])

    if len(list_objects) > 0:
        if len(list_objects) == 1 and list_objects[0]['Key'] == object_key:
            # It's a single file
            local_filepath = Path(destination)
            local_filepath.parent.mkdir(parents=True, exist_ok=True)
            if local_filepath.exists():
                print(f"File already exists: {local_filepath}. Skipping download.")
            else:
                s3.download_file(bucket_name, object_key, str(local_filepath))
                print(f"Successfully downloaded: {object_key} to {local_filepath}")
        else:
            # It's a folder or has auxiliary files
            for obj in list_objects:
                if obj["Key"].endswith(".tif"):
                    local_filepath = Path(destination)
                    local_filepath.parent.mkdir(parents=True, exist_ok=True)
                    if local_filepath.exists():
                        print(f"File already exists: {local_filepath}. Skipping download.")
                    else:
                        s3.download_file(bucket_name, obj["Key"], str(local_filepath))
                        print(f"Successfully downloaded: {obj['Key']} to {local_filepath}")
    else:
        raise FileNotFoundError("Source is empty! Nothing to download.")


def upload_files_to_s3(s3_url: str, local_directory: str, upload_list: list):
    """
    Upload specified files from a directory to an S3 bucket and then delete the local directory.

    :param s3_url: The S3 URL where the files should be uploaded.
    :type s3_url: str
    :param local_directory: The local directory containing the files to upload.
    :type local_directory: str
    :param upload_list: List of filenames to upload.
    :type upload_list: list
    """
    # Check if local directory exists
    if not os.path.exists(local_directory):
        raise FileNotFoundError(f"The local directory {local_directory} does not exist.")

    s3 = boto3.client('s3')
    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    s3_path = parsed_url.path.lstrip('/')
    
    # Configure the transfer for multipart upload and threading
    config = TransferConfig(
        multipart_threshold=1024 * 25,  # 25MB
        max_concurrency=10,
        multipart_chunksize=1024 * 25,  # 25MB
        use_threads=True
    )

    # Upload the specified files
    for filename in upload_list:
        for root, dirs, files in os.walk(local_directory):
            if filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_key = os.path.join(s3_path, relative_path).replace("\\", "/")

                # Print what is being uploaded and where
                print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")

                # Upload the file
                s3.upload_file(local_path, bucket_name, s3_key, Config=config)
                print(f"Successfully uploaded: {local_path} to s3://{bucket_name}/{s3_key}")



def read_json_file_from_s3(s3_url):
    fs = s3fs.S3FileSystem(anon=False)
    # s3_path = f's3://{bucket_name}/{s3_key}'
    try:
        with fs.open(s3_url, 'r') as f:
            data = json.load(f)
            print(f"Successfully read JSON data from {s3_url}")
            return data
    except Exception as e:
        print(f"Error reading {s3_url}: {e}")
        return None

def read_csv_file_from_s3(s3_url):
    fs = s3fs.S3FileSystem(anon=False)
    try:
        with fs.open(s3_url, 'r') as f:
            data = pd.read_csv(f, dtype="float32")
            print(f"Successfully read CSV data from {s3_url}")
            return data
    except Exception as e:
        print(f"Error reading {s3_url}: {e}")
        return None

def read_tif_file_from_s3(s3_url):
    fs = s3fs.S3FileSystem(anon=False)
    # s3_path = f's3://{bucket_name}/{s3_key}'
    try:
        with fs.open(s3_url, 'rb') as f:
            data = rioxarray.open_rasterio(f)
            print(f"Successfully read TIF data from {s3_url}")
            return data
    except Exception as e:
        print(f"Error reading {s3_url}: {e}")
        return None

def extract_data(json_file_path):

    print (f'Extracting data from {json_file_path}')
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # data = read_json_file_from_s3(json_file_path)
    
    wavelengths = data['waves'].split(',')
    fwhms = data['fwhm'].split(',')
    
    return wavelengths, fwhms

def write_data_to_file(out_file_path, wavelengths, fwhms):
    print (f'Writing data to {out_file_path}')
    with open(out_file_path, "w") as file:
        for index, (wl, fwhm) in enumerate(zip(wavelengths, fwhms)):
            wl_um = float(wl)  # Convert string to float
            fwhm_um = float(fwhm)  # Convert string to float
            file.write(f"{index}\t{wl_um:.4f}\t{fwhm_um:.4f}\n")


def convert_csv_to_envi(s3_key, local_path):
    # get files from s3
    # download_from_s3(s3_key, local_path)

    # Read in CSV file and convert to BIP ENVI file
    # csv_path = sys.argv[1]
    output_path = local_path.replace(".csv", "")
    output_hdr_path = output_path + ".hdr"

    # Check if the output file already exists
    if Path(output_path).exists() and Path(output_hdr_path).exists():
        print(f"ENVI files already exist: {output_path} and {output_hdr_path}. Skipping process.")
        return
    # else:
    #     print(f"Processing CSV to ENVI: {local_path}")

    # spectra_df = pd.read_csv(local_path, dtype="float32")
    spectra_df = read_csv_file_from_s3(s3_key)
    spectra_df = spectra_df.fillna(-9999)
    lines, bands = spectra_df.shape
    wavelengths = spectra_df.keys().values
    hdr = {
        "lines": str(lines),
        "samples": "1",
        "bands": str(bands),
        "header offset": "0",
        "file type": "ENVI Standard",
        "data type": "4",
        "interleave": "bip",
        "byte order": "0",
        "data ignore value": "-9999",
        "wavelength": wavelengths
    }
    out_file = envi.create_image(output_hdr_path, hdr, ext='', force=True)
    output_mm = out_file.open_memmap(interleave='source', writable=True)

    # Iterate through dataframe and write to output memmap
    for index, row in spectra_df.iterrows():
        output_mm[index, 0, :] = row.values
    # Write to disk
    del output_mm
    print(f"Successfully converted CSV to ENVI: {output_path} and {output_hdr_path}")


def calculate_slope(dem, cell_size=1):
    # Calculate gradients in the x and y directions
    # np.gradient returns the gradient with respect to each axis (y-axis, x-axis)
    gy, gx = np.gradient(dem, cell_size)

    # Calculate slope
    # Slope is calculated as the hypotenuse of the gradient components
    # We use np.arctan to get the angle in radians and then convert to degrees
    slope = np.arctan(np.sqrt(gx**2 + gy**2)) * (180 / np.pi)

    return slope

def calculate_aspect(dem):
    # Calculate gradients in the x and y directions
    # np.gradient returns the gradient with respect to each axis (y-axis, x-axis)
    gy, gx = np.gradient(dem)

    # Calculate aspect
    # Aspect is the arctangent of the opposite (gx) over the adjacent (gy)
    # We convert from radians to degrees and adjust the direction
    aspect = np.degrees(np.arctan2(-gx, gy))

    # Adjust aspect values to make sure they are between 0 and 360 degrees
    # North is 0 degrees, so a north-facing slope has an aspect of 0
    aspect = (aspect + 360) % 360

    return aspect

def l1b_preprocess(conf): 

    print ('Preprocessing image for input rdn, loc, and obs datacubes.')

    paths = [conf.input_radiance, conf.input_loc, conf.input_obs]
    if not all(os.path.exists(path) for path in paths):
        print("Not all required datacubes exist. Building new ones...")

        # read waves and fwhm
        wave_data = pd.read_csv(conf.wavelength_path, sep='\t', header=None)
        waves = wave_data.iloc[:,1].values
        fwhm = wave_data.iloc[:,2].values

        # get image bounding box/extents
        # 0: latitude, 1: longitude, 2: geo elev, 3: sea lvl elev
        data = rioxarray.open_rasterio(conf.loc_filepath_local)
        latitude = data[0].data
        longitude = data[1].data
        elevation = data[3].data

        lat_mean = np.nanmean(latitude)
        lon_mean = np.nanmean(longitude)

        # map info
        transform = data.rio.transform()
        map_info = '{Geographic Lat/Lon, 1.0, 1.0, %0.10f, %0.10f, %0.10f, %0.10f, WGS-84, units=degrees}' % (transform.c, transform.f, transform.a, -1*transform.e)
        # map_info =  f'{{Geographic Lat/Lon, 1, 1, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5]*-1},WGS-84, units=degrees}}'

        # Raster dimensions
        height, width = data.shape[1], data.shape[2]  # Notice shape indexing depends on raster being 2D or 3D

        # Calculate corner coordinates
        corner_1 = transform * (0, 0)  # Top Left
        corner_2 = transform * (width, 0)  # Top Right
        corner_3 = transform * (width, height)  # Bottom Right
        corner_4 = transform * (0, height)  # Bottom Left

        # crs
        crs=data.rio.crs
        crs_py = CRS(crs)
        coordinate_system_string = crs_py.to_wkt()

        # Correct the order and organize into a list for the bounding box in the header
        bounding_box = [list(corner_1), list(corner_2), list(corner_3), list(corner_4)]

        # Output the bounding box for verification
        # print("Bounding Box:", bounding_box)

        latitude[np.isnan(latitude)] = -9999
        longitude[np.isnan(longitude)] = -9999
        elevation[np.isnan(elevation)] = -9999

        altitude_m = 490 * 1000
        pathlength = altitude_m - elevation

        acquisition_time = conf.acquisition_date
        start_time = dt.datetime.strptime(acquisition_time, '%Y-%m-%dT%H:%M:%SZ')
        start_time = start_time.replace(tzinfo=dt.timezone.utc)

        # create radiance datacube and hdr file
        raster = rioxarray.open_rasterio(conf.l1b_filepath_local)
        raster = raster.data
        # convert units for isofit
        raster = raster / 10

        mask = raster[1].astype(float)
        mask = mask==mask[0][0]

        lines = raster.shape[1]
        samples = raster.shape[2]

        # write radiance header
        print ('Writing radiance datacube and hdr file.')
        rad_header = envi_header_dict()
        rad_header['description']= 'Radiance datacube: micro-watts/cm^2/nm/sr'
        rad_header['lines']= lines
        rad_header['samples']= samples
        rad_header['bands']= len(waves)
        rad_header['wavelength'] = waves 
        rad_header['fwhm'] = fwhm
        rad_header['interleave']= 'bil'
        rad_header['data type'] = 4
        rad_header['data ignore value'] = -9999
        rad_header['byte order'] = 0
        rad_header['wavelength units'] = "nanometers"
        rad_header['map info'] = map_info
        rad_header['start acquisition time'] = start_time
        rad_header['coordinate system string'] = coordinate_system_string
        rad_header['bounding box'] = bounding_box
        rad_header['sensor'] = 'TD1'

        # write radiance cube
        writer = WriteENVI(conf.input_radiance, rad_header)
        for line_num in range(lines):
            line = raster[:,line_num,:].astype(float)
            line = line.T
            line[mask[line_num]] =-9999
            writer.write_line(line,line_num)
        del raster


        # create location datacube and hdr file
        print ('Writing location datacube and hdr file.')
        loc_header = envi_header_dict()
        loc_header['description']= 'Location datacube'
        loc_header['lines']= lines
        loc_header['samples']= samples
        loc_header['bands']= 3
        loc_header['interleave']= 'bil'
        loc_header['data type'] = 4
        loc_header['data ignore value'] = -9999
        loc_header['band_names'] = ['Longitude', 'Latitude','Elevation']
        loc_header['byte order'] = 0
        loc_header['map info'] = map_info
        loc_header['start acquisition time'] = start_time
        loc_header['coordinate system string'] = coordinate_system_string
        loc_header['bounding box'] = bounding_box
        loc_header['Longitude'] = '(WGS-84)'
        loc_header['Latitude'] = '(WGS-84)'
        loc_header['Elevation'] = '(m)'
        loc_header['sensor'] = 'TD1'

        writer = WriteENVI(conf.input_loc,loc_header)
        writer.write_band(longitude,0)
        writer.write_band(latitude,1)
        writer.write_band(elevation,2)

        # create observation datacube and hdr file
        print ('Writing observation datacube and hdr file.')
        # load geometry files
        data = rioxarray.open_rasterio(conf.sol_filepath_local)
        solar_az = data[0].data
        solar_zn = data[1].data
        solar_zn = 90 - solar_zn
        data = rioxarray.open_rasterio(conf.obs_filepath_local)
        sensor_az = data[0].data
        sensor_zn = data[1].data
        sensor_zn = 90 - sensor_zn

        # convert azimuth to 0-360
        # if np.any(solar_az < 0):
        #     solar_az += 180
        #     solar_az = solar_az % 360
        # if np.any(sensor_az < 0):
        #     solar_az += 180
        #     solar_az = solar_az % 360

        slope = calculate_slope(elevation)
        aspect = calculate_aspect(elevation)

        cosine_i = calc_cosine_i(np.radians(solar_zn),
                                np.radians(solar_az),
                                np.radians(slope),
                                np.radians(aspect))

        rel_az = np.radians(solar_az-sensor_az)
        phase =  np.arccos(np.cos(np.radians(solar_zn)))*np.cos(np.radians(solar_zn))
        phase += np.sin(np.radians(solar_zn))*np.sin(np.radians(solar_zn))*np.cos(rel_az)

        # Use timezonefinder to find the timezone of the given location
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=lat_mean, lng=lon_mean)

        # If timezone_str is None, the location is likely in international waters or there is no data available
        if timezone_str is None:
            raise ValueError("Could not determine the timezone for the given location.")

        # Use zoneinfo to attach the local timezone to the acquisition time
        local_time_zone = ZoneInfo(timezone_str)
        acquisition_time_obj = start_time.replace(tzinfo=local_time_zone)

        # Convert local acquisition time to UTC
        utc_time = acquisition_time_obj.astimezone(ZoneInfo('UTC'))

        # Calculate the hours and fraction of hours from midnight to start_time
        utc_time = utc_time.hour + utc_time.minute / 60 + utc_time.second / 3600
        utc_time*= np.ones(latitude.shape)


        # mask out products
        solar_az[mask] = -9999
        solar_zn[mask] = -9999
        sensor_az[mask] = -9999
        sensor_zn[mask] = -9999
        slope[mask] = -9999
        aspect[mask] = -9999
        cosine_i[mask] = -9999
        rel_az[mask] = -9999
        phase[mask] = -9999
        utc_time[mask] = -9999
        pathlength[mask] = -9999

        # write observation datacube and hdr file
        obs_header = envi_header_dict()
        obs_header['description']= 'Observation datacube'
        obs_header['lines']= lines
        obs_header['samples']= samples
        obs_header['data ignore value'] = -9999
        obs_header['bands']= 10
        obs_header['interleave']= 'bil'
        obs_header['data type'] = 4
        obs_header['byte order'] = 0
        obs_header['band_names'] = ['path length', 'to-sensor azimuth',
                                    'to-sensor zenith','to-sun azimuth',
                                        'to-sun zenith','phase', 'slope',
                                        'aspect', 'cosine i','UTC time']
        obs_header['map info'] = map_info
        obs_header['start acquisition time'] = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        obs_header['start acquisition time'] = start_time
        obs_header['coordinate system string'] = coordinate_system_string
        obs_header['bounding box'] = bounding_box
        obs_header['sensor'] = 'TD1'

        writer = WriteENVI(conf.input_obs,obs_header)
        writer.write_band(pathlength,0)
        writer.write_band(sensor_az,1)
        writer.write_band(sensor_zn,2)
        writer.write_band(solar_az,3)
        writer.write_band(solar_zn,4)
        writer.write_band(phase,5)
        writer.write_band(slope,6)
        writer.write_band(aspect,7)
        writer.write_band(cosine_i,8)
        writer.write_band(utc_time,9)
        print ('Finished preprocessing.')

    else:
        print ('All required datacubes exist. Skipping preprocessing.')
