FROM  tensorflow/tensorflow:devel-gpu

# Enviorment setup
ENV CONDA_PATH=/opt/anaconda3
ENV ENVIRONMENT_NAME=main
SHELL ["/bin/bash", "-c"]

# Updates
RUN apt-get update
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN apt-get install libxrender1 -y
RUN apt-get install -y libsm6 libxext6 -y
RUN apt-get install -y libxrender-dev -y

# Download and install Anaconda.
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2019.07-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2019.07-Linux-x86_64.sh -b -p ${CONDA_PATH}"

# Initializes Conda for bash shell interaction.
RUN ${CONDA_PATH}/bin/conda init bash

# Upgrade Conda to the latest version
RUN ${CONDA_PATH}/bin/conda update -n base -c defaults conda -y

# Create the work environment and setup its activation on start.
RUN ${CONDA_PATH}/bin/conda create --name ${ENVIRONMENT_NAME} anaconda -y
RUN echo conda activate ${ENVIRONMENT_NAME} >> /root/.bashrc

# Install libraries
RUN . ${CONDA_PATH}/bin/activate ${ENVIRONMENT_NAME} && conda install -c conda-forge keras 

RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install tensorflow-gpu
RUN pip3 install rdkit-pypi
RUN apt install libgfortran4 -y 
