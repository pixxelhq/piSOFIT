FROM rayproject/ray:2.4.0-py310-aarch64

USER root

# Remove the outdated kubernetes-xenial repository
RUN rm -f /etc/apt/sources.list.d/kubernetes.list

# Add the correct Kubernetes repository
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' > /etc/apt/sources.list.d/kubernetes.list

# Install essential packages
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      gfortran \
      make \
      unzip

USER ray
WORKDIR /home/ray

# Copy the pixxel_isofit directory into the container
COPY --chown=ray:users . pixxel_isofit/

# Create the conda environment using the environment file
RUN conda update conda &&\
    conda config --prepend channels conda-forge &&\
    conda env create --file pixxel_isofit/recipe/pixxel_isofit_environment.yml &&\
    echo "conda activate isofit_env" >> ~/.bashrc
ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD"


# Set environment variables for ISOFIT if needed (example)
# RUN echo "export VARIABLE_NAME=DIRECTORY" >> ~/.bashrc

# Run the scripts to install 6S and sRTMnet
RUN pixxel_isofit/scripts/download-and-build-6s.sh
RUN pixxel_isofit/scripts/download-and-unpack-sRTMnet.sh

# Set environment variables for the installed software
ENV SIXS_DIR="/home/ray/6sv-2.1"
ENV EMULATOR_PATH="/home/ray/sRTMnet_v100/sRTMnet_v100"
ENV MODTRAN_DIR=""

# Explicitly set the shell to bash so the Jupyter server defaults to it
ENV SHELL=/bin/bash

# Start a bash shell
CMD ["/bin/bash", "-c", "source ~/.bashrc && bash"]
