# Pixxel ISOFIT (piSOFIT)

ISOFIT stands for Imaging Spectrometer Optimal Fitting. This version of ISOFIT is built specifically for Pixxel satellites, thus now named, piSOFIT (pie-so-fit). It is an open-source tool designed to perform atmospheric correction and surface reflectance retrieval from imaging spectroscopy data. The primary goal of ISOFIT is to accurately retrieve the surface reflectance by modeling the interaction of light between the Earth's surface and the atmosphere. The core isofit correction is maintained by NASA JPL, while piSOFIT is developed and maintained by Jeremy Kravitz (jeremy@pixxel.space).  

## General Procedure
#### 1. Inputs

- Surface spectral libraries: Spectral libraries including diverse sets of surface reflectances for surface priors.
- Image files: L1b radiance, elevation, solar/sensor geometries, latitude/longitude
- Image metadata: wavelengths, FWHM, acquisition time, sensor altitude
- Instrument model: Currently using spectrally invariant SNR of 20

#### 2. Preparation and preprocessing

- Surface model: Multi-component surface model constructed using Guassian PDFs of surface spectral libraries
- Image file to ENVI format: L1b radiance, elevation, solar/sensor geometries, latitude/longitude into ENVI binary and header files

#### 3. Superpixel Segmentation 

- Segment Image: Apply SLIC segmentation algorithm to the image data to reduce the computational load by grouping pixels with similar spectral characteristics based on PCA.
- Representative Spectra Selection: For each superpixel, select representative spectra, which will be used in subsequent steps for parameter estimation.

#### 4. Forward Modeling

- Simulation Execution: Run the radiative transfer model to simulate the at-sensor radiance based on the initial estimates of atmospheric and surface conditions.
- Comparison with Observed Data: Compare the simulated radiance to the actual observed data from the imaging spectrometer.
- This version uses 6SV1 and a MODTRAN emulator called sRTMnet built by JPL

#### 5. Water Vapor Pre-solve

- Broad Range Simulations: Perform initial simulations using the radiative transfer model across a wide range of water vapor values, focusing on moisture-sensitive spectral bands.
- Refine Water Vapor Range: Analyze these simulations to identify and narrow down the most probable water vapor values for more detailed and accurate inversion modeling.

#### 6. Look-Up-Table (LUT) Generation

- Variable Selection: Identify the range of key variables (AOT550, water vapor) relevant to the observed scene.
- Grid Definition: Define a grid over the selected variable range, specifying discrete values each variable can take. The granularity of the grid impacts the accuracy and computational efficiency.
- Populate LUT: Conduct detailed radiative transfer simulations for each combination of parameters in the grid and store the results in the LUT.

#### 7. Optimal Estimation Loop

- Cost Function Evaluation: Compute a cost function that quantifies the difference between the observed radiance and the model-predicted radiance, incorporating penalties from prior knowledge (see Thompson et al., 2018).
- Jacobian Calculation: Compute the Jacobian matrix, which contains partial derivatives of the radiance with respect to each parameter, to understand how changes in parameters affect the observed radiance.
- Parameter Updating: Use an optimization algorithm (Currently uses Truncated Newton Conjugate-Gradient Descent.) to update the parameter estimates to minimize the cost function.
- Iteration: Repeat the simulation, evaluation, and updating steps iteratively until the change in the cost function is below a predefined threshold, indicating convergence.

#### 8. Uncertainty Analysis

- Error Propagation: Estimate how uncertainties in inputs affect the retrieved surface reflectance using statistical techniques.
- Covariance Assessment: Analyze covariance matrices to quantify uncertainties and dependencies among retrieved parameters.

#### 9. Empirical Line Method (ELM)

- Linear Regression: Perform linear regression between superpixel inverted reflectances and the corresponding at-sensor radiances for each superpixel. This regression provides coefficients that adjust the model output to match the observed radiance precisely, effectively tailoring the model to the specific scene.
- Extrapolation: Using superpixel regression coefficients, extrapolate to retrieve surface reflectance for each individual pixel.

## Current Operational Procedure

This version of piSOFIT is designed to integrate with Pixxel's AWS S3 bucket architecture. A Dockerfile is provided to build a local working container. To run the piSOFIT routine inside the container you will need pixxel AWS access. From inside the pixxel_isofit directory run:

1. docker build -t iso_image .
2. docker run -it --name pixxel_isofit_container \
    -e AWS_ACCESS_KEY_ID=XXXXXXXXXXXX \
    -e AWS_SECRET_ACCESS_KEY= XXXXXXXXXXXX \
    iso_image

This will build and run the docker container with isofit and all of its dependencies, including 6SV1 and sRTMnet, and open a terminal from inside the container. To run the atmospheric correction for a particular image, navigate to inside the pixxel_isofit directory and run:

1. python test.py <s3/L0/dir/> <s3/L1b/dir/>

All necessasry files will be downloaded to a local directory and processed within the container and final outputs will be uploaded to <s3/L0/dir/isofit/...>


## Link to info
https://pixxel.atlassian.net/wiki/spaces/IPR/pages/1093140514/IsoFit
