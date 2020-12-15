# ezmock

A Python interface to the EZmock code.
This package does two things:
1. Automate the process of creating EZmocks on an HPC cluster managed with Slurm.
2. Easily load created EZmocks as Python objects for convenient manipulation in (say) Jupyter notebooks.


## Installation

You should already have a compiled EZmock binary somewhere.
Make sure you have execute permissions for the binary.

You should also have a environment variable `$SCRATCH` defined on your system,
giving the path to the your scratch directory.
Then clone this repo,
enter the conda environment you want to install inside some conda environment,
and install via `pip`:
```bash
pip install -e .
```
This step should install all necessary dependencies.

Next, edit the hardcoded paths `PLANCK15PK_PATH` and `EZMOCK_BINARY_PATH` in `ezmock/ezmock_job.py`
to point to your linear PK file
and your EZmock binary.
These are used as defaults in the code.

Then, edit the Slurm batch script template in the `templates` directory to fit
your Slurm needs.
In particular, make sure to edit the following options:
* `--output` and `--error` should point to your desired Slurm output paths
* `--cpus-per-task` (on NERSC, for example, it's often better to have this be _twice_ the number of OpenMP threads)
* `--partition`
* `--account`
* `OMP_NUM_THREADS`

`ezmock` will require a `$SCRATCH/ezmock` directory,
where it stores all relevant files,
including EZmock catalogs.


## Example usage

Please refer to the example notebook.
