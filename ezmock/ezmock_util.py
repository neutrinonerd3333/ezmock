import copy
import datetime
import fnmatch
import os
import os.path
import pickle
import random
import shlex
import shutil
import subprocess
import time

import numpy as np

import jinja2
import nbodykit.algorithms
import nbodykit.cosmology
import nbodykit.source.catalog


# TODO make these globals configurable

scratch_path = os.environ['SCRATCH']
EZMOCK_OUT_DIR = os.path.join(scratch_path, 'ezmock', 'mock_output/')  # this slash is important
# EZmock doesn't sanitize input :<
os.makedirs(EZMOCK_OUT_DIR, exist_ok=True)

# create a directory for temp files if it doesn't yet exist
EZMOCK_TEMP_DIR = os.path.join(scratch_path, 'ezmock', 'tempfiles/')
os.makedirs(EZMOCK_TEMP_DIR, exist_ok=True)


PLANCK15PK_PATH = '~/EZmock_eBOSS_LRG_ELG_empty/pks-and-corrs/20191026-planck15-loguniform-pk.dat'
EZMOCK_BINARY_PATH = '~/EZmock_eBOSS_LRG_ELG_empty/fortran/pro_EZmock_pk_cf'


def fortran_repr(obj):
    if type(obj) is bool:
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
    TODO
    """
    params = dict()

    if output_path[-1] != '/':
        raise ValueError('specified output_path must explicitly end with a slash "/"')
        # EZmock will otherwise treat the last directory as part of a filename :<

    params['datafile_path'] = output_path
    params['datafile_prefix'] = output_prefix
    params['boxsize'] = boxsize
    params['grid_num'] = grid_num
    
    z = redshift
    params['redshift'] = z

    linear_pk_obj = nbodykit.cosmology.LinearPower(cosmology, z)
    sigma8_z0 = linear_pk_obj.sigma8
    sigma8_z = linear_pk_obj.sigma_r(8)
    growth_factor = sigma8_z / sigma8_z0
    params['grow2z0'] = growth_factor**2
    
    # f * H(z) * a/h = f * H/(1+z)/h
    growth_rate = cosmology.scale_independent_growth_rate(z)  # 'f'
    efunc = cosmology.efunc(z)  # This is H(z) / (100 * h) in units of km/s/Mpc
    scale_factor = 1 / (1 + z)
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
    params['pknwfile'] = '~/EZmock_eBOSS_LRG_ELG_empty/pks-and-corrs/PlanckDM.nowiggle.pk'

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
    # recognize existing output_prefixes
    # TODO check matching of params if we've already down the EZmock
    seen_output_prefixes = [fname[:-4] for fname in fnmatch.filter(os.listdir(EZMOCK_OUT_DIR), '*.dat')]
    return output_prefix in seen_output_prefixes


class EZmockJob():
    _current_dir = os.path.dirname(os.path.realpath(__file__))
    _template_dir = os.path.join(_current_dir, '../templates')
    jinja_loader = jinja2.FileSystemLoader(_template_dir)  # jinja wants abs path
    jinja_template_env = jinja2.Environment(loader=jinja_loader)
    ezmock_multisubmit_template = jinja_template_env.get_template('ezmock-multisubmit-template.sbatch')
    
    def __init__(self, time_limit, filenames):
        """
        filenames : list of tuples (params_file, output_params_file, params_pickle, output_params_pickle)
        """
        self.time_limit = time_limit
        self.filenames = copy.deepcopy(filenames)
    
    def run(self, verbose=True):
        cls.ezmock_multisubmit_template.render(
            job_name='ezmock',
            time_limit=cls._slurm_time_format(self.time_limit),
            ezmock_binary=EZMOCK_BINARY_PATH,
            filenames=self.filenames,
        )
        
        with open(sbatch_path, 'w+') as fp:
            fp.write(generated_sbatch)
        os.chmod(sbatch_path, 0o755)
        
        if verbose:
            print('Generated sbatch at {}'.format(sbatch_path))

        # run EZmock
        ezmock_cmd = ['sbatch', shlex.quote(sbatch_path)] if sbatch else [shlex.quote(sbatch_path)]
        print(' '.join(ezmock_cmd))
        
        process = subprocess.run(ezmock_cmd)
        if process.returncode != 0:
            raise RuntimeError('EZmock returned with nonzero exit code')

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
        time_str : str
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


class EZmock():
    """
    A Python wrapper around individual EZmock instances.

    Parameters
    ----------
    """
    
    _current_dir = os.path.dirname(os.path.realpath(__file__))
    _template_dir = os.path.join(_current_dir, '../templates')
    jinja_loader = jinja2.FileSystemLoader(_template_dir)  # jinja wants abs path
    jinja_template_env = jinja2.Environment(loader=jinja_loader)
    ezmock_submit_template = jinja_template_env.get_template('ezmock-submit-template.sbatch')
    
    @staticmethod
    def _compute_time_limit(box_side_length, catalog_size, dilute_factor):
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
        
        We round this up to 20 minutes and scale as described, but require
        time limits above a minimum of 20 minutes.
        
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
        
        time_limit_2pcf = datetime.timedelta(minutes=20) \
            * dilute_factor**2 \
            * number_ratio**2 \
            * side_length_ratio**(-3)
        MIN_TIME = datetime.timedelta(minutes=20)
        return max(MIN_TIME, time_limit_2pcf)
        
    
    @classmethod
    def generate(
        cls,
        output_prefix,
        verbose=False,
        sbatch=False,
        ezmock_binary=EZMOCK_BINARY_PATH,
        **kwargs,
    ):
        if does_output_prefix_exist(output_prefix):
            raise UserWarning(f'output_prefix {output_prefix} already exists!')
            return
        
        params, coparams = generate_ezmock_params(output_prefix, **kwargs)
        
        # timestamp we use for all generated files
        # not unique because of daylight savings time, hmmm
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # use this filename to guarantee uniqueness (jobs might be submitted within a second!)
        temp_filename = f'{timestamp}-{output_prefix}'

        # make params file for EZmock
        params_filename = f'{temp_filename}.ini'
        params_file_path = os.path.join(EZMOCK_TEMP_DIR, params_filename)
        generate_ezmock_input_file(params, params_file_path)
        if verbose:
            print(f'Generated params file at {params_file_path}')
            
        # create shell script for EZmock
        sbatch_filename = f'{temp_filename}.sbatch'
        sbatch_path = os.path.join(EZMOCK_TEMP_DIR, sbatch_filename)

        time_limit = cls._compute_time_limit(
            params['boxsize'],
            params['expect_sum_pdf'],
            params['dilute_factor']
        )

        cls._generate_ezmock_sbatch(
            sbatch_path,
            params_file_path,
            time_limit,
            ezmock_binary=ezmock_binary,
        )
        if verbose:
            print('Generated sbatch at {}'.format(sbatch_path))

        # run EZmock
        ezmock_cmd = ['sbatch', shlex.quote(sbatch_path)] if sbatch else [shlex.quote(sbatch_path)]
        print(' '.join(ezmock_cmd))

        # make sure subprocess stdout comes out!
        process_stdout = None if verbose else subprocess.DEVNULL
        process = subprocess.run(ezmock_cmd, stdout=process_stdout)

        if process.returncode != 0:
            raise RuntimeError('EZmock returned with nonzero exit code')

        # copy the params file into the output directory
        # eventually make this part of fortran code
        output_params_file_name = output_prefix + '.ini'
        output_params_file_path = os.path.join(EZMOCK_OUT_DIR, output_params_file_name)
        shutil.copyfile(params_file_path, output_params_file_path)
        
        params_pickle = os.path.join(EZMOCK_OUT_DIR, output_prefix + '.pickle')
        with open(params_pickle, 'wb') as fp:
            pickle.dump(coparams, fp)
    
    
    def __init__(self, output_prefix):
        self.name = output_prefix
        with open(os.path.join(EZMOCK_OUT_DIR, output_prefix + '.pickle'), 'rb') as fp:
            self.params = pickle.load(fp)
        self._import_ezmock_output(self.params['compute_CF'], self.params['compute_CF_zdist'])

        
    def _import_ezmock_output(self, import_cf_real, import_cf_zdist):
        """
        """
        self.catalog = self._import_ezmock_catalog()
        
        # TODO refactor this
        
        pk_path = os.path.join(EZMOCK_OUT_DIR, f'{self.name}.dat.CICassign.NGPbin.pk.mono')
        pk_real = np.genfromtxt(pk_path, names=['k', 'power'])
        
        pk_path = os.path.join(EZMOCK_OUT_DIR, f'{self.name}.dat.CICassign.NGPbin.pk.zdist.mono')
        pk_zdist_mono = np.genfromtxt(pk_path, names=['k', 'power'])
        
        pk_path = os.path.join(EZMOCK_OUT_DIR, f'{self.name}.dat.CICassign.NGPbin.pk.zdist.quad')
        pk_zdist_quad = np.genfromtxt(pk_path, names=['k', 'power'])
        
        self.ezmock_pk = {
            'real': pk_real,
            'zdist': {
                'mono': pk_zdist_mono,
                'quad': pk_zdist_quad,
            },
        }
        
        # nbodykit computed
        nmesh = self.params['grid_num']
        mesh_kwargs = dict(
            Nmesh=nmesh,
            compensated=True,
            interlaced=True,
            resampler='cic',
        )

        realmesh = self.catalog.to_mesh(position='Position', **mesh_kwargs)
        pk_real = nbodykit.algorithms.FFTPower(realmesh, mode='1d', dk=0.005, kmin=0)

        zdistmesh = self.catalog.to_mesh(position='RSDPosition', **mesh_kwargs)
        pk_zdist = nbodykit.algorithms.FFTPower(
            zdistmesh,
            mode='2d',
            dk=0.005,
            kmin=0,
            Nmu=5,
            los=[0,0,1],
            poles=[0,2]
        )

        self.pk = {
            'real': pk_real,
            'zdist': pk_zdist,
        }
        
        self.twop_cf = None
        if import_cf_real or import_cf_zdist:
            self.twop_cf = dict()
        if import_cf_real:
            cf_file = '{}.dat.bin5.corr.mono'.format(self.name)
            cf_path = os.path.join(EZMOCK_OUT_DIR, cf_file)
            self.twop_cf['real'] = np.genfromtxt(cf_path, names=['r', 'corr'])
        if import_cf_zdist:
            self.twop_cf['zdist'] = dict()
            
            cf_file = '{}.dat.bin5.corr.zdist.mono'.format(self.name)
            cf_path = os.path.join(EZMOCK_OUT_DIR, cf_file)
            self.twop_cf['zdist']['mono'] = np.genfromtxt(cf_path, names=['r', 'corr'])
            
            cf_file = '{}.dat.bin5.corr.zdist.quad'.format(self.name)
            cf_path = os.path.join(EZMOCK_OUT_DIR, cf_file)
            self.twop_cf['zdist']['quad'] = np.genfromtxt(cf_path, names=['r', 'corr'])
      
    
    def _import_ezmock_catalog(self, **kwargs):
        """
        Import a catalog generated by EZmock with given `name_stem`
        """
        fname = os.path.join(EZMOCK_OUT_DIR, f'{self.name}.dat')
        column_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        catalog = nbodykit.source.catalog.file.CSVCatalog(fname, column_names)

        catalog['Position'] = catalog['x'][:, None] * [1, 0, 0] + \
            catalog['y'][:, None] * [0, 1, 0] + catalog['z'][:, None] * [0, 0, 1]
        catalog['Velocity'] = catalog['vx'][:, None] * [1, 0, 0] + \
            catalog['vy'][:, None] * [0, 1, 0] + catalog['vz'][:, None] * [0, 0, 1]

        # see nbodykit docs for meaning
        catalog['VelocityOffset'] = self.params['rsd_factor'] * catalog['Velocity']

        line_of_sight = [0, 0, 1]
        catalog['RSDPosition'] = catalog['Position'] + catalog['VelocityOffset'] * line_of_sight

        catalog.attrs['BoxSize']= [self.params['boxsize']]*3

        catalog.los = line_of_sight
        for key, val in kwargs.items():
            catalog.attrs[key] = val

        return catalog

    @classmethod
    def _generate_ezmock_sbatch(
        cls,
        sbatch_path,
        params_file_path,
        time_limit,
        ezmock_binary=EZMOCK_BINARY_PATH,
    ):
        """
        Parameters
        ----------
        sbatch_path : str
        params_file_path : str
        time_limit : datetime.timedelta
            time limit for the slurm job
        ezmock_binary : str
            path to the EZmock binary
        """
        time_limit_str = cls._slurm_time_format(time_limit)
        print('Time limit = {}'.format(time_limit_str))
    
        generated_sbatch = cls.ezmock_submit_template.render(
            job_name='ezmock',
            time_limit=time_limit_str,
            params_file_path=params_file_path,
            ezmock_binary=ezmock_binary,
        )
        with open(sbatch_path, 'w+') as fp:
            fp.write(generated_sbatch)
        os.chmod(sbatch_path, 0o755)
        
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
        time_str : str
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
        
