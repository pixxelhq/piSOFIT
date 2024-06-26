import logging
import os
import subprocess
from datetime import datetime
from os.path import exists, join
from types import SimpleNamespace
from warnings import warn

import click
import numpy as np
import ray
from scipy.io import loadmat
from spectral.io import envi

import isofit.utils.template_construction as tmpl
from isofit.core import isofit
from isofit.core.common import envi_header
from isofit.utils import analytical_line, empirical_line, extractions, segment


def cli_apply_oe(debug_args, profile, **kwargs):
    """Apply OE to a block of data"""

    if debug_args:
        click.echo("Arguments to be passed:")
        for key, value in kwargs.items():
            click.echo(f"  {key} = {value!r}")
    else:
        if profile:
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()

        # SimpleNamespace converts a dict into dot-notational
        apply_oe(SimpleNamespace(**kwargs))

        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.dump_stats(profile)

    click.echo("Done")

#%
def apply_oe(args):

    use_superpixels = args.empirical_line or args.analytical_line

    # ray.shutdown()
    ray.init(
        num_cpus=args.n_cores,
        _temp_dir=args.ray_temp_dir,
        include_dashboard=False,
        local_mode=args.n_cores == 1,
    )

    if args.sensor[:3] != "NA-":
        errstr = 'Argument error: sensor must be in format "NA-YYYYMMDD"'
        raise ValueError(errstr)

    if args.num_neighbors is not None and len(args.num_neighbors) > 1:
        if not args.analytical_line:
            raise ValueError(
                "If num_neighbors has multiple elements, --analytical_line must be True"
            )
        if args.empirical_line:
            raise ValueError(
                "If num_neighbors has multiple elements, only --analytical_line is valid"
            )

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s || %(filename)s:%(funcName)s() | %(message)s",
        level=args.logging_level,
        filename=args.log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.info(args)

    logging.info("Checking input data files...")
    rdn_dataset = envi.open(envi_header(args.input_radiance))
    rdn_size = (rdn_dataset.shape[0], rdn_dataset.shape[1])
    del rdn_dataset
    for infile_name, infile in zip(
        ["input_radiance", "input_loc", "input_obs"],
        [args.input_radiance, args.input_loc, args.input_obs],
    ):
        if os.path.isfile(infile) is False:
            err_str = (
                f"Input argument {infile_name} give as: {infile}.  File not found on"
                " system."
            )
            raise ValueError("argument " + err_str)
        if infile_name != "input_radiance":
            input_dataset = envi.open(envi_header(infile), infile)
            input_size = (input_dataset.shape[0], input_dataset.shape[1])
            if not (input_size[0] == rdn_size[0] and input_size[1] == rdn_size[1]):
                err_str = (
                    f"Input file: {infile_name} size is {input_size}, which does not"
                    f" match input_radiance size: {rdn_size}"
                )
                raise ValueError(err_str)
    logging.info("...Data file checks complete")


    lut_params = tmpl.LUTConfig(emulator=args.emulator_base)

    # logging.info("Setting up files and directories....")
    # paths = tmpl.Pathnames(args)
    # paths.make_directories()
    # paths.stage_files()
    logging.info("Files and directories already setu up")


    # Based on the sensor type, get appropriate year/month/day info from initial condition.
    # We'll adjust for line length and UTC day overrun later
    if args.sensor[:3] == "NA-":
        dt = datetime.strptime(args.sensor[3:], "%Y%m%d")
    else:
        raise ValueError(
            "Datetime object could not be obtained. Please check file name of input"
            " data."
        )

    dayofyear = dt.timetuple().tm_yday


    # get obs metadata
    (
        h_m_s,
        day_increment,
        mean_path_km,
        mean_to_sensor_azimuth,
        mean_to_sensor_zenith,
        mean_to_sun_zenith,
        mean_relative_azimuth,
        valid,
        to_sensor_zenith_lut_grid,
        to_sun_zenith_lut_grid,
        relative_azimuth_lut_grid,
    ) = tmpl.get_metadata_from_obs(args.obs_working_path, lut_params)

    # overwrite the time in case original obs has an error in that band
    if h_m_s[0] != dt.hour and h_m_s[0] >= 24:
        h_m_s[0] = dt.hour
        logging.info(
            "UTC hour did not match start time minute. Adjusting to that value."
        )
    if h_m_s[1] != dt.minute and h_m_s[1] >= 60:
        h_m_s[1] = dt.minute
        logging.info(
            "UTC minute did not match start time minute. Adjusting to that value."
        )

    if day_increment:
        dayofyear += 1

    gmtime = float(h_m_s[0] + h_m_s[1] / 60.0)

    # get radiance file, wavelengths, fwhm
    radiance_dataset = envi.open(envi_header(args.radiance_working_path))
    wl_ds = np.array([float(w) for w in radiance_dataset.metadata["wavelength"]])
    if args.wavelength_path:
        if os.path.isfile(args.wavelength_path):
            chn, wl, fwhm = np.loadtxt(args.wavelength_path).T
            if len(chn) != len(wl_ds) or not np.all(np.isclose(wl, wl_ds, atol=0.01)):
                raise ValueError(
                    "Number of channels or center wavelengths provided in wavelength file do not match"
                    " wavelengths in radiance cube. Please adjust your wavelength file."
                )
        else:
            pass
    else:
        logging.info(
            "No wavelength file provided. Obtaining wavelength grid from ENVI header of radiance cube."
        )
        wl = wl_ds
        if "fwhm" in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata["fwhm"]])
        elif "FWHM" in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata["FWHM"]])
        else:
            fwhm = np.ones(wl.shape) * (wl[1] - wl[0])

    # Close out radiance dataset to avoid potential confusion
    del radiance_dataset


    # check wavelength grid of surface file
    if args.surface_path:
        model_dict = loadmat(args.surface_path)
        wl_surface = model_dict["wl"][0]
        if len(wl_surface) != len(wl):
            raise ValueError(
                "Number of channels provided in surface model file does not match"
                " wavelengths in radiance cube. Please rebuild your surface model."
            )
        if not np.all(np.isclose(wl_surface, wl, atol=0.01)):
            logging.warning(
                "Center wavelengths provided in surface model file do not match"
                " wavelengths in radiance cube. Please consider rebuilding your"
                " surface model for optimal performance."
            )

    # Convert to microns if needed
    if wl[0] > 100:
        logging.info("Wavelength units of nm inferred...converting to microns")
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # write wavelength file
    wl_data = np.concatenate(
        [np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis], fwhm[:, np.newaxis]],
        axis=1,
    )
    np.savetxt(args.wavelength_path, wl_data, delimiter=" ")

    (
        mean_latitude,
        mean_longitude,
        mean_elevation_km,
        elevation_lut_grid,
    ) = tmpl.get_metadata_from_loc(
        args.loc_working_path, lut_params, pressure_elevation=args.pressure_elevation
    )

    #%

    if args.emulator_base is not None:
        if elevation_lut_grid is not None and np.any(elevation_lut_grid < 0):
            to_rem = elevation_lut_grid[elevation_lut_grid < 0].copy()
            elevation_lut_grid[elevation_lut_grid < 0] = 0
            elevation_lut_grid = np.unique(elevation_lut_grid)
            if len(elevation_lut_grid) == 1:
                elevation_lut_grid = None
                mean_elevation_km = elevation_lut_grid[
                    0
                ]  # should be 0, but just in case
            logging.info(
                "Scene contains target lut grid elements < 0 km, and uses 6s (via"
                " sRTMnet).  6s does not support targets below sea level in km units. "
                f" Setting grid points {to_rem} to 0."
            )
        if mean_elevation_km < 0:
            mean_elevation_km = 0
            logging.info(
                f"Scene contains a mean target elevation < 0.  6s does not support"
                f" targets below sea level in km units.  Setting mean elevation to 0."
            )

    # Need a 180 - here, as this is already in MODTRAN convention
    mean_altitude_km = (
        mean_elevation_km
        + np.cos(np.deg2rad(180 - mean_to_sensor_zenith)) * mean_path_km
    )

    logging.info("Observation means:")
    logging.info(f"Path (km): {mean_path_km}")
    logging.info(f"To-sensor azimuth (deg): {mean_to_sensor_azimuth}")
    logging.info(f"To-sensor zenith (deg): {mean_to_sensor_zenith}")
    logging.info(f"To-sun zenith (deg): {mean_to_sun_zenith}")
    logging.info(f"Relative to-sun azimuth (deg): {mean_relative_azimuth}")
    logging.info(f"Altitude (km): {mean_altitude_km}")

    if args.emulator_base is not None and mean_altitude_km > 99:
        logging.info(
            "Adjusting altitude to 99 km for integration with 6S, because emulator is"
            " chosen."
        )
        mean_altitude_km = 99

    # We will use the model discrepancy with covariance OR uncorrelated
    # Calibration error, but not both.
    if args.model_discrepancy_path is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = args.uncorrelated_radiometric_uncertainty


    # Superpixel segmentation
    if use_superpixels:
        if not exists(args.lbl_working_path) or not exists(
            args.radiance_working_path
        ):
            logging.info("Segmenting...")
            segment(
                spectra=(args.radiance_working_path, args.lbl_working_path),
                nodata_value=-9999,
                npca=args.n_pca,
                segsize=args.segmentation_size,
                nchunk=args.chunksize,
                n_cores=args.n_cores,
                loglevel=args.logging_level,
                logfile=args.log_file,
            )

        # Extract input data per segment
        for inp, outp in [
            (args.radiance_working_path, args.rdn_subs_path),
            (args.obs_working_path, args.obs_subs_path),
            (args.loc_working_path, args.loc_subs_path),
        ]:
            if not exists(outp):
                logging.info("Extracting " + outp)
                extractions(
                    inputfile=inp,
                    labels=args.lbl_working_path,
                    output=outp,
                    chunksize=args.chunksize,
                    flag=-9999,
                    n_cores=args.n_cores,
                    loglevel=args.logging_level,
                    logfile=args.log_file,
                )

    if args.presolve:
        # write modtran presolve template
        tmpl.write_modtran_template(
            atmosphere_type=args.atmosphere_type,
            fid=args.fid,
            altitude_km=mean_altitude_km,
            dayofyear=dayofyear,
            to_sun_zenith=mean_to_sun_zenith,
            to_sensor_azimuth=mean_to_sensor_azimuth,
            to_sensor_zenith=mean_to_sensor_zenith,
            relative_azimuth=mean_relative_azimuth,
            gmtime=gmtime,
            elevation_km=mean_elevation_km,
            output_file=args.h2o_template_path,
            ihaze_type="AER_NONE",
        )

        if args.emulator_base is None and args.prebuilt_lut is None:
            max_water = tmpl.calc_modtran_max_water(args)
        else:
            max_water = 6

        # run H2O grid as necessary
        if not exists(envi_header(args.h2o_subs_path)) or not exists(
            args.h2o_subs_path
        ):
            # Write the presolve connfiguration file
            h2o_grid = np.linspace(0.01, max_water - 0.01, 10).round(2)
            logging.info(f"Pre-solve H2O grid: {h2o_grid}")
            logging.info("Writing H2O pre-solve configuration file.")
            tmpl.build_presolve_config(
                paths=args,
                h2o_lut_grid=h2o_grid,
                n_cores=args.n_cores,
                use_emp_line=use_superpixels,
                surface_category=args.surface_category,
                emulator_base=args.emulator_base,
                uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
                prebuilt_lut_path=args.prebuilt_lut,
            )

            # Run modtran retrieval
            logging.info("Run ISOFIT initial guess")
            retrieval_h2o = isofit.Isofit(
                args.h2o_config_path,
                level="INFO",
                logfile=args.log_file,
            )
            retrieval_h2o.run()
            del retrieval_h2o

            # clean up unneeded storage
            if args.emulator_base is None:
                for to_rm in args.rtm_cleanup_list:
                    cmd = "rm " + join(args.lut_h2o_directory, to_rm)
                    logging.info(cmd)
                    subprocess.call(cmd, shell=True)
        else:
            logging.info("Existing h2o-presolve solutions found, using those.")

        h2o = envi.open(envi_header(args.h2o_subs_path))
        h2o_est = h2o.read_band(-1)[:].flatten()

        p05 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 2)
        p95 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 98)
        margin = (p95 - p05) * 0.5

        lut_params.h2o_range[0] = max(lut_params.h2o_min, p05 - margin)
        lut_params.h2o_range[1] = min(max_water, max(lut_params.h2o_min, p95 + margin))

    h2o_lut_grid = lut_params.get_grid(
        lut_params.h2o_range[0],
        lut_params.h2o_range[1],
        lut_params.h2o_spacing,
        lut_params.h2o_spacing_min,
    )

    logging.info("Full (non-aerosol) LUTs:")
    logging.info(f"Elevation: {elevation_lut_grid}")
    logging.info(f"To-sensor zenith: {to_sensor_zenith_lut_grid}")
    logging.info(f"To-sun zenith: {to_sun_zenith_lut_grid}")
    logging.info(f"Relative to-sun azimuth: {relative_azimuth_lut_grid}")
    logging.info(f"H2O Vapor: {h2o_lut_grid}")


    logging.info(args.state_subs_path)
    if (
        not exists(args.state_subs_path)
        or not exists(args.uncert_subs_path)
        or not exists(args.rfl_subs_path)
    ):
        tmpl.write_modtran_template(
            atmosphere_type=args.atmosphere_type,
            fid=args.fid,
            altitude_km=mean_altitude_km,
            dayofyear=dayofyear,
            to_sun_zenith=mean_to_sun_zenith,
            to_sensor_azimuth=mean_to_sensor_azimuth,
            to_sensor_zenith=mean_to_sensor_zenith,
            relative_azimuth=mean_relative_azimuth,
            gmtime=gmtime,
            elevation_km=mean_elevation_km,
            output_file=args.modtran_template_path,
        )

        logging.info("Writing main configuration file.")
        tmpl.build_main_config(
            paths=args,
            lut_params=lut_params,
            h2o_lut_grid=h2o_lut_grid,
            elevation_lut_grid=(
                elevation_lut_grid
                if elevation_lut_grid is not None
                else [mean_elevation_km]
            ),
            to_sensor_zenith_lut_grid=(
                to_sensor_zenith_lut_grid
                if to_sensor_zenith_lut_grid is not None
                else [mean_to_sensor_zenith]
            ),
            to_sun_zenith_lut_grid=(
                to_sun_zenith_lut_grid
                if to_sun_zenith_lut_grid is not None
                else [mean_to_sun_zenith]
            ),
            relative_azimuth_lut_grid=(
                relative_azimuth_lut_grid
                if relative_azimuth_lut_grid is not None
                else [mean_relative_azimuth]
            ),
            mean_latitude=mean_latitude,
            mean_longitude=mean_longitude,
            dt=dt,
            use_emp_line=use_superpixels,
            n_cores=args.n_cores,
            surface_category=args.surface_category,
            emulator_base=args.emulator_base,
            uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
            multiple_restarts=args.multiple_restarts,
            segmentation_size=args.segmentation_size,
            pressure_elevation=args.pressure_elevation,
            prebuilt_lut_path=args.prebuilt_lut,
        )

        # Run modtran retrieval
        logging.info("Running ISOFIT with full LUT")
        retrieval_full = isofit.Isofit(
            args.isofit_full_config_path, level="INFO", logfile=args.log_file
        )
        retrieval_full.run()
        del retrieval_full

        # clean up unneeded storage
        if args.emulator_base is None:
            for to_rm in args.rtm_cleanup_list:
                cmd = "rm " + join(args.full_lut_directory, to_rm)
                logging.info(cmd)
                subprocess.call(cmd, shell=True)


    if not exists(args.rfl_working_path) or not exists(args.uncert_working_path):
        # Determine the number of neighbors to use.  Provides backwards stability and works
        # well with defaults, but is arbitrary
        if not args.num_neighbors:
            nneighbors = [int(round(3950 / 9 - 35 / 36 * args.segmentation_size))]
        else:
            nneighbors = [n for n in args.num_neighbors]

        if args.empirical_line:
            # Empirical line
            logging.info("Empirical line inference")
            empirical_line(
                reference_radiance_file=args.rdn_subs_path,
                reference_reflectance_file=args.rfl_subs_path,
                reference_uncertainty_file=args.uncert_subs_path,
                reference_locations_file=args.loc_subs_path,
                segmentation_file=args.lbl_working_path,
                input_radiance_file=args.radiance_working_path,
                input_locations_file=args.loc_working_path,
                output_reflectance_file=args.rfl_working_path,
                output_uncertainty_file=args.uncert_working_path,
                isofit_config=args.isofit_full_config_path,
                nneighbors=nneighbors[0],
                n_cores=args.n_cores,
            )
        elif args.analytical_line:
            logging.info("Analytical line inference")
            analytical_line(
                args.radiance_working_path,
                args.loc_working_path,
                args.obs_working_path,
                args.working_directory,
                output_rfl_file=args.rfl_working_path,
                output_unc_file=args.uncert_working_path,
                loglevel=args.logging_level,
                logfile=args.log_file,
                n_atm_neighbors=nneighbors,
                n_cores=args.n_cores,
                smoothing_sigma=args.atm_sigma,
            )

    logging.info("Done.")
    ray.shutdown()
