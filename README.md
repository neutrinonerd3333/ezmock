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

Next, edit the hardcoded path `EZMOCK_BINARY_PATH` in `ezmock/ezmock_job.py`
to point to your EZmock binary.
If you want to compute bispectra with this code,
build Cheng Zhao's `bispec` script and edit `BISPEC_BINARY` in `ezmock/ezmock_job.py`.

Then, create a Slurm batch script template `ezmock-multisubmit-template.sbatch`
in the `templates` directory to fit your Slurm needs.
There are some examples in the directory; you should basically only need to modify the slurm options.
Make sure to edit the following options:
* `--output` and `--error` should point to your desired Slurm output paths
* `--cpus-per-task` (on NERSC, for example, it's often better to have this be _twice_ the number of OpenMP threads)
* `--partition`
* `--account`
* `OMP_NUM_THREADS`

`ezmock` will require a `$SCRATCH/ezmock` directory,
where it stores all relevant files,
including EZmock catalogs.


## Example usage

Please refer to the example notebook (`docs/tutorial.ipynb`).


## Contributing

Pull requests are most welcome,
as this code grew organically from some personal scripts
and is by no means perfect.
