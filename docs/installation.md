# Installation

### step 1: CATX installation
The simplest way to install CATX is through PyPI:
`pip install catx`

We recommend using either Docker, Singularity, or conda to use the repository.

### step 2 [Optional]: JAX GPU installation
CATX installation in step 1 uses JAX on CPU.

To unleash the full speed of CATX and have access to a GPU, a GPU version of JAX must be installed.

JAX installation is different depending on your CUDA version. Follow these [instructions](https://github.com/google/jax#installation)
to install JAX with the relevant accelerator support.

TL;DR:

run `pip install --upgrade pip` then run one of the following depending on your machine:

- No GPU (not needed as JAX cpu was installed in step 1):

    `pip install --upgrade "jax[cpu]"`

- GPU with CUDA 11 and cuDNN 8.2 or newer:

    `pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`


- GPU with Cuda >= 11.1 and cudnn >= 8.0.5:

    `pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`



## Installation for development:

- Clone your fork of this repo:

    `git clone git@github.com:instadeepai/catx.git`

- Go to directory: `cd catx`

- Add upstream repo

    `git remote add upstream https://github.com/instadeepai/catx.git`

- Create your venv or conda environment and install dependencies:

    conda:
    ```
    conda env create -f environment.yaml
    conda activate catx
    ```

    venv:
    ```
    python3 -m venv
    pip install -e .[tool,test]
    ```

- \[Optional\] follow step 2 above for JAX to use GPU.
