# Use the official Python image as a base
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .


# Create the conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Set the environment path
ENV PATH /opt/conda/envs/maxi_env/bin:$PATH


# Activate the conda environment
SHELL ["conda", "run", "-n", "maxi_env", "/bin/bash", "-c"]

RUN conda install -c conda-forge mamba && \
    mamba install -c conda-forge autogluon=0.8.2 && \
    mkdir -m 700 flagged


# https://github.com/gradio-app/gradio/issues/3693#issuecomment-1745577523
#!mkdir -m 700 flagged

# Copy the current directory contents into the container at /app
COPY . /app
ADD learner.pkl /app/learner.pkl
ADD predictor.pkl /app/predictor.pkl
ADD models /app/models
#ADD utils /app/utils
ADD metadata.json /app/metadata.json

COPY . .

EXPOSE 8080

# Specify the command to run on container start
CMD ["python", "app.py"]
#CMD ["/bin/bash"]
