"""
File for functions related to EZmock generation.
"""

import copy
import datetime
import fnmatch
import os
import os.path
import pickle
import random
import shlex
import subprocess
import time

import jinja2
import nbodykit.cosmology


# TODO make these globals configurable

scratch_path = os.environ['SCRATCH']
EZMOCK_OUT_DIR = os.path.join(scratch_path, 'ezmock', 'mock_output/')  # this slash is important
# EZmock doesn't sanitize input :<
os.makedirs(EZMOCK_OUT_DIR, exist_ok=True)

# create a directory for temp files if it doesn't yet exist
EZMOCK_TEMP_DIR = os.path.join(scratch_path, 'ezmock', 'tempfiles/')
os.makedirs(EZMOCK_TEMP_DIR, exist_ok=True)

PLANCK15PK_PATH = \
    '~/ezmock-codez/EZmock_eBOSS_LRG_ELG_empty/pks-and-corrs/20191026-planck15-loguniform-pk.dat'
EZMOCK_BINARY_PATH = '~/ezmock-codez/EZmock_eBOSS_LRG_ELG_empty/fortran/pro_EZmock_pk_cf'


def fortran_repr(obj):
    """
    Parameters
    ----------
    obj : any
        The object to represent in Fortran.

    Returns : str
        A string representation for Fortran. (In pracitce, '.true.' and '.false.' for bools
        and repr(obj) for everything else.)
    -------
    """
    if isinstance(obj, bool):
        return '.true.' if obj else '.false.'
    return repr(obj)


EZMOCK_PARAM_NAMES = frozenset([
    'datafile_path',
    'datafile_prefix',
    'boxsize',
    'grid_num',
    'redshift',
    'grow2z0',
    'expect_sum_pdf',
    'expect_A_pdf',
    'density_cut',
    'scatter2',
    'zdist_rate',
    'zdist_fog',
    'iseed',
    'density_sat',
    'modify_pk',
    'modify_pdf',
    'scatter',
    'use_whitenoise_file',
    'whitenoise_file',
    'pkfile',
    'antidamping',
    'pknwfile',
    'compute_CF',
    'compute_CF_zdist',
    'dilute_factor',
    'skiplines',
    'max_r',
    'bin_size',
    'om',
    'twod_corr_suffix',
])


def generate_ezmock_params(
    output_prefix,
    output_path=EZMOCK_OUT_DIR,
    pk_file_path=PLANCK15PK_PATH,
    boxsize=1000,
    grid_num=256,
    redshift=0.9873,
    density=0.00064,
    expect_A_pdf=0.25,
    density_cut=0,
    scatter2=1,
    zdist_fog=150,
    iseed=None,
    dilute_factor=None,
    compute_CF=False,
    compute_CF_zdist=False,
    cf_max_r=250,
    cf_bin_size=5,
    cosmology=nbodykit.cosmology.Planck15,
):
    """
    Parameters
    ----------
    output_prefix : str
        name of this EZmock; will be included in all related files
    output_path : str, optional
        path to the directory to put the final EZmock results
    pk_file_path : str, optional
        path to the linear power spectrum file to use.
        Optional as long as you make sure you edit the default
        to point to the correct location...
    boxsize : float
        side length of the box, in Mpc/h
    grid_num : int
        number of mesh cells in one dimension (box side length / mesh side length)
    redshift : float
        the redshift of the catalog, used when computing redshift-space statistics
    density : float
        desired number density of the catalog, in 1 / (Mpc/h)^3
    expect_A_pdf : float
        PDF slope for EZmock.
    density_cut : float
        Density threshold for EZmock, in units of objects per cell.
        Must be nonnegative.
    scatter2 : float
        Scatter parameter for EZmock.
    zdist_fog : float
        Must be nonnegative.
    iseed : int32, optional
        A random seed to give to EZmock; generated automatically if not specified.
    dilute_factor : float, optional
        Fraction of objects in EZmock catalog to use to compute the correlation function.
        Required if `compute_CF or compute_CF_zdist`.
    compute_CF : bool, optional
        whether to compute the correlation function in real space (default: no)
    compute_CF_zdist : bool, optional
        whether to compute the correlation funciton in redshift space (default: no)
    cf_max_r : float
        maximum distance to compute the correlation function to, in Mpc/h
    cf_bin_size : float
        size of bins for correlation function computation, in Mpc/h
    cosmology : nbodykit.cosmology.cosmology.Cosmology
        the cosmology to use

    Return
    ------
    params : dict
        The params dictionary that will be input to the EZmock binary.
    coparams : dict
        The params dictionary, with some additional data for the EZmock class.
    """
    params = dict()

    if output_path[-1] != '/':
        raise ValueError('specified output_path must explicitly end with a slash "/"')
        # EZmock will otherwise treat the last directory as part of a filename :<

    params['datafile_path'] = output_path
    params['datafile_prefix'] = output_prefix
    params['boxsize'] = boxsize
    params['grid_num'] = grid_num

    params['redshift'] = redshift

    linear_pk_obj = nbodykit.cosmology.LinearPower(cosmology, redshift)
    sigma8_z0 = linear_pk_obj.sigma8
    sigma8_z = linear_pk_obj.sigma_r(8)
    growth_factor = sigma8_z / sigma8_z0
    params['grow2z0'] = growth_factor**2

    # f * H(z) * a/h = f * H/(1+z)/h
    growth_rate = cosmology.scale_independent_growth_rate(redshift)  # 'f'
    efunc = cosmology.efunc(redshift)  # This is H(z) / (100 * h) in units of km/s/Mpc
    scale_factor = 1 / (1 + redshift)
    rsd_factor = 1 / (scale_factor * 100 * efunc)
    params['zdist_rate'] = growth_rate / rsd_factor

    params['expect_sum_pdf'] = int(density * boxsize**3)
    params['expect_A_pdf'] = expect_A_pdf
    params['density_cut'] = density_cut
    params['scatter2'] = scatter2
    params['zdist_fog'] = zdist_fog

    if iseed is None:
        iseed = random.randrange(-2**31, 2**31)
    params['iseed'] = iseed

    # not used, but need in the file
    params['density_sat'] = 100


    # tilt pk at small scales. Typical value is about 0 to 1
    params['modify_pk'] = 0

    # tune the tail of pk. Typical value is -0.1 < modify_pdf < 0.1
    params['modify_pdf'] = 0

    # fixed
    params['scatter'] = 10

    # has no effect but must be included
    params['use_whitenoise_file'] = False
    params['whitenoise_file'] = '/home2/chuang/data/BigMD_BDM3p5_and_white_noise/BigMD_WhiteNoise/BigMD_960_wn_delta'

    params['pkfile'] = pk_file_path

    # tunes BAO signal, but has no effect if >= 1
    params['antidamping'] = 2

    # not used if antidamping > 1
    params['pknwfile'] = '~/ezmock-codez/EZmock_eBOSS_LRG_ELG_empty/pks-and-corrs/PlanckDM.nowiggle.pk'

    # correlation function computation

    if (dilute_factor is None) and (compute_CF or compute_CF_zdist):
        raise ValueError('Must specify numerical dilute_factor if computing correlation functions.')
    if dilute_factor is not None and not 0 <= dilute_factor <= 1:
        raise ValueError('dilute_factor must be between 0 and 1.')

    params['compute_CF'] = compute_CF
    params['compute_CF_zdist'] = compute_CF_zdist

    if dilute_factor is None:
        dilute_factor = 0.1337
    params['dilute_factor'] = dilute_factor

    params['skiplines'] = 0
    params['max_r'] = cf_max_r
    params['bin_size'] = cf_bin_size
    params['om'] = cosmology.Omega0_m
    params['twod_corr_suffix'] = '.bin5.corr'

    # TODO figure out a better way to do this
    coparams = copy.deepcopy(params)
    coparams['rsd_factor'] = rsd_factor
    coparams['cosmology'] = cosmology

    return (params, coparams)


def generate_ezmock_input_file(params, path):
    """
    Given the EZmock parameter dictionary `params`,
    create an EZmock input file at `path` with these parameters.

    Parameters
    ----------
    params : dict
        dictionary of EZmock parameters, must specify all EZmock params
        (and no others).
    path : string
        absolute path to write EZmock input file
    """
    if not os.path.isabs(path):
        raise ValueError('path specification must be absolute!')

    param_keys = frozenset(params.keys())
    if not param_keys <= EZMOCK_PARAM_NAMES:
        raise ValueError('Unrecognized EZmock params: {}'.format(param_keys - EZMOCK_PARAM_NAMES))
    if not param_keys >= EZMOCK_PARAM_NAMES:
        raise ValueError('Unspecified EZmock params: {}'.format(EZMOCK_PARAM_NAMES - param_keys))

    header = ' &EZmock_v0_input\n'
    param_file_lines = ['{} = {}\n'.format(var_name, fortran_repr(var_val)) for var_name, var_val in params.items()]
    meat = ''.join(param_file_lines)
    footer = '/\n'
    params_string = header + meat + footer

    with open(path, 'w+') as fp:
        fp.write(params_string)


def does_output_prefix_exist(output_prefix):
    """
    Check whether a given `output_prefix` exists.

    Parameters
    ----------
    output_prefix : str
        A candidate identifier string for an EZmock

    Returns
    -------
    bool
        Whether the identifier string is already used by an EZmock in the default EZmock directory.
    """
    # recognize existing output_prefixes
    # TODO check matching of params if we've already down the EZmock
    seen_output_prefixes = [fname[:-4] for fname in fnmatch.filter(os.listdir(EZMOCK_OUT_DIR), '*.dat')]
    return output_prefix in seen_output_prefixes


def _compute_time_limit(box_side_length, catalog_size, max_r, dilute_factor):
    """
    Compute a reasonable (if somewhat conservative) time limit for a Slurm
    job for a single EZmock. This is based on the fact that we're bottle-
    necked by computation of the two-point function, which scales
    - linearly with volume
    - quadratic with density of objects

    We can re-express this in terms of parameter values we know (box side
    length L, number of objects N, and dilute_factor) to get scaling

    dilute^2 * N^2 / L^3.

    For a 5.12m catalog on a 2 Gpc/h box with no dilution, we needed 12 min.

    We round this up to 16 minutes and scale as described, but require
    time limits above a minimum of 8 minutes.

    Parameters
    ----------
    box_side_length : number
        Side length of the simulation box in Mpc/h
    catalog_size : number
        Number of objects in the catalog
    dilute_factor : number
        Dilution factor

    Returns
    -------
    time_limit : datetime.timedelta
        Computed time limit based on above scaling.
    """
    number_ratio = catalog_size / 5120000
    side_length_ratio = box_side_length / 2000
    max_r_ratio = max_r / 250

    time_limit_2pcf = datetime.timedelta(minutes=16) \
        * dilute_factor**2 \
        * number_ratio**2 \
        * side_length_ratio**(-3) \
        * max_r_ratio**3
    MIN_TIME = datetime.timedelta(minutes=12)
    return max(MIN_TIME, time_limit_2pcf)


def make_job(
    output_prefix,
    verbose=False,
    ezmock_binary=EZMOCK_BINARY_PATH,
    **kwargs,
):
    """
    Parameters
    ----------
    output_prefix : str
        Identifier for the EZmock, to be used in all relevant files.
    verbose : bool, optional
    kwargs : dict
        Keyword arguments per `generate_ezmock_params`

    Returns
    -------
    EZmockJob
        An object containing all relevant job information that can be run or
        chunked with other `EZmockJob` objects.

    See Also
    --------
    generate_ezmock_params :
        generates a params dictionary containing EZmock parameters
    """
    if does_output_prefix_exist(output_prefix):
        raise UserWarning(f'output_prefix {output_prefix} already exists!')

    params, coparams = generate_ezmock_params(output_prefix, **kwargs)

    # timestamp we use for all generated files
    # not unique because of daylight savings time, hmmm
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    # use this filename to guarantee uniqueness
    # this might be called again within a second!
    temp_filename = f'{timestamp}-{output_prefix}'

    # make params file for EZmock
    params_filename = f'{temp_filename}.ini'
    params_file_path = os.path.join(EZMOCK_TEMP_DIR, params_filename)
    generate_ezmock_input_file(params, params_file_path)
    if verbose:
        print(f'Generated params file at {params_file_path}')

    time_limit = _compute_time_limit(
        params['boxsize'],
        params['expect_sum_pdf'],
        params['dilute_factor'],
        params['max_r'],
    )

    pickle_filename = f'{temp_filename}.pickle'
    pickle_file_path = os.path.join(EZMOCK_TEMP_DIR, pickle_filename)
    with open(pickle_file_path, 'wb') as fp:
        pickle.dump(coparams, fp)


    # copy the params file into the output directory
    # eventually make this part of fortran code
    output_params_file_name = output_prefix + '.ini'
    output_params_file_path = os.path.join(EZMOCK_OUT_DIR, output_params_file_name)

    output_pickle_file_path = os.path.join(EZMOCK_OUT_DIR, output_prefix + '.pickle')

    filenames_tuple = (
        ezmock_binary,
        params_file_path,
        output_params_file_path,
        pickle_file_path,
        output_pickle_file_path,
    )
    return EZmockJob(time_limit, [filenames_tuple])


class EZmockJob():
    """
    Object representing an EZmock job to send to Slurm.
    """

    _current_dir = os.path.dirname(os.path.realpath(__file__))
    _template_dir = os.path.join(_current_dir, '../templates')
    jinja_loader = jinja2.FileSystemLoader(_template_dir)  # jinja wants abs path
    jinja_template_env = jinja2.Environment(loader=jinja_loader)
    ezmock_multisubmit_template = jinja_template_env.get_template('ezmock-multisubmit-template.sbatch')

    @classmethod
    def chunk(cls, jobs, chunk_size):
        """
        Given a list of individual EZmock jobs, produce a series of EZmock jobs
        by chunking them into segments of `chunk_size` jobs.

        Parameters
        ----------
        jobs : list of EZmockJob
            The jobs to divide into chunks.
        chunk_size : int
            Number of jobs in each chunk.

        Returns
        -------
        job_chunks : list of EZmockJob
            A list of the chunked jobs, each consisting of `chunk_size` jobs
            from `jobs` (except possibly the last).
        """
        job_chunks = []
        for i in range(0, len(jobs), chunk_size):
            job_chunk = cls.empty()
            for job in jobs[i:i+chunk_size]:
                job_chunk += job
            job_chunks.append(job_chunk)

        return job_chunks

    @classmethod
    def empty(cls):
        """
        Returns an empty EZmock job.
        """
        return EZmockJob(datetime.timedelta(), [])

    def __init__(self, time_limit, filenames):
        """
        Parameters
        ----------
        time_limit : datetime.timedelta
            time limit to use on the Slurm job.
        filenames : list of 5-tuples of strings
            Each tuple consists of relevant paths. These paths are
            `binary_file` : path to the EZmock binary to use
            `params_file` : path to the params file to feed to EZmock
            `output_params_file` : path to copy the params file to
            `params_pickle` : path to a pickled version of the params
            `output_params_pickle` : path to copy the params pickle
        """
        self.time_limit = time_limit
        self.filenames = copy.deepcopy(filenames)

    def run(self, verbose=True, time_limit=None):
        """
        Run the EZmock job.
        """
        actual_time_limit = self.time_limit if time_limit is None else time_limit
        time_limit_str = self._slurm_time_format(actual_time_limit)

        # TODO shell-escape filenames!
        generated_sbatch = self.ezmock_multisubmit_template.render(
            job_name='ezmock',
            time_limit=time_limit_str,
            filenames=self.filenames,
        )

        # microsecond field because we can run these quite quickly
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        sbatch_path = os.path.join(EZMOCK_TEMP_DIR, f'{timestamp}.sbatch')
        with open(sbatch_path, 'w+') as fp:
            fp.write(generated_sbatch)
        os.chmod(sbatch_path, 0o755)

        if verbose:
            print(f'Generated sbatch at {sbatch_path}')

        # run EZmock
        ezmock_cmd = ['sbatch', shlex.quote(sbatch_path)]
        print(' '.join(ezmock_cmd))

        return subprocess.run(ezmock_cmd, check=True)

    def __add__(self, other):
        return EZmockJob(
            self.time_limit + other.time_limit,
            self.filenames + other.filenames,
        )

    @staticmethod
    def _slurm_time_format(timedelta):
        """
        Format the duration `timedelta` in the Slurm time limit format,
        as specified in the sbatch(1) man page. Rounds down to the nearest
        second.

        Parameters
        ----------
        timedelta : datetime.timedelta
            Duration to format.

        Returns
        -------
        str
            Slurm-formatted string.
        """
        DAY = datetime.timedelta(days=1)
        SECOND = datetime.timedelta(seconds=1)
        days, remainder = divmod(timedelta, DAY)
        total_seconds = remainder // SECOND
        rounded_remainder = datetime.timedelta(seconds=total_seconds)

        if days == 0:
            time_str = str(rounded_remainder)
        else:
            time_str = '{}-{}'.format(days, rounded_remainder)

        return time_str
