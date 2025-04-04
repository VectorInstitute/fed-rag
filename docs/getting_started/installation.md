# Installation

## Installing from package managers

### PyPi

As seen in the previous quickstart examples, we can install FedRAG via `pip`:

``` sh
pip install fed-rag
```

### Conda

For `conda` users, `fed-rag` has been published to the
[`conda-forge`](https://conda-forge.org/) channel, and thus can be installed
with `conda` using the below command:

``` sh
conda install -c conda-forge fed-rag
```

## Installing from source

To install from source, first clone the repository:

``` sh
# https
git clone https://github.com/VectorInstitute/fed-rag.git

# ssh
git clone git@github.com:VectorInstitute/fed-rag.git
```

After cloning the repository, you have a few options for installing the library.
The next two subsections outline how to complete the installation using either
`pip` or `uv`, respectively.

### Using `pip`

To complete the installation, first `cd` into the `fed-rag` directory and then
run the following `pip install` command:

``` sh
cd fed-rag
pip install -e .
```

!!! tip
    We recommended to always use a fresh virtual environment for new projects.
    Before running the above command, ensure that your dedicated virtual environment
    is active.

### Using `uv`
