import datetime
import fnmatch
import os
import os.path
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


scratch_path = os.environ['SCRATCH']
EZMOCK_OUT_DIR = os.path.join(scratch_path, 'ezmock', 'mock_output/')  # this slash is important
# EZmock doesn't sanitize input :<
os.makedirs(EZMOCK_OUT_DIR, exist_ok=True)

# create a directory for temp files if it doesn't yet exist
EZMOCK_TEMP_DIR = os.path.join(scratch_path, 'ezmock', 'tempfiles/')
os.makedirs(EZMOCK_TEMP_DIR, exist_ok=True)


PLANCK15PK_PATH = '/home/users/txz/EZmock_eBOSS_LRG_ELG_empty/pks-and-corrs/20191026-planck15-loguniform-pk.dat'

EZMOCK_BINARY_PATH = '/home/users/txz/EZmock_eBOSS_LRG_ELG_empty/fortran/pro_EZmock_pk_cf'


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
    zdist_rate=74.07, # !f*H(z)*a/h = f*H/(1+z)/h = 0.84426*109.403/1.84/0.6777
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
    
    params['redshift'] = redshift
    2
    linear_pk_obj = nbodykit.cosmology.LinearPower(cosmology, redshift)
    sigma8_z0 = linear_pk_obj.sigma8
    sigma8_z = linear_pk_obj.sigma_r(8)
    growth_factor = sigma8_z / sigma8_z0
    params['grow2z0'] = growth_factor**2
    
    params['expect_sum_pdf'] = int(density * boxsize**3)
    params['expect_A_pdf'] = expect_A_pdf
    params['density_cut'] = density_cut
    params['scatter2'] = scatter2
    params['zdist_rate'] = zdist_rate
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
    params['om'] = 0.3089  # from Planck 2015, to match UNIT sim omega_matter
    params['twod_corr_suffix'] = '.bin5.corr'

    return params


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
        

def import_ezmock_catalog(name_stem, **kwargs):
    """
    Import a catalog generated by EZmock with given `name_stem`
    
    TODO
    """
    fname = os.path.join(EZMOCK_OUT_DIR, "{}.dat".format(name_stem))
    column_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    catalog = nbodykit.source.catalog.file.CSVCatalog(fname, column_names)
    catalog['Position'] = catalog['x'][:, None] * [1, 0, 0] + \
        catalog['y'][:, None] * [0, 1, 0] + catalog['z'][:, None] * [0, 0, 1]
    catalog['Velocity'] = catalog['vx'][:, None] * [1, 0, 0] + \
        catalog['vy'][:, None] * [0, 1, 0] + catalog['vz'][:, None] * [0, 0, 1]
    
    for key, val in kwargs.items():
        catalog.attrs[key] = val

    return catalog


def does_output_prefix_exist(output_prefix):
    # recognize existing output_prefixes
    # TODO check matching of params if we've already down the EZmock
    seen_output_prefixes = [fname[:-4] for fname in fnmatch.filter(os.listdir(EZMOCK_OUT_DIR), '*.dat')]
    return output_prefix in seen_output_prefixes


class EZmock():
    """
    A Python wrapper around individual EZmock instances.

    Parameters
    ----------
    """
    
    # TODO don't hardcode! this
    jinja_loader = jinja2.FileSystemLoader('/home/users/txz/ezmock-templates')
    jinja_template_env = jinja2.Environment(loader=jinja_loader)
    ezmock_submit_template = jinja_template_env.get_template('ezmock-submit-template.sbatch')
    
    
    def __init__(
        self,
        output_prefix,
        verbose=False,
        sbatch=False,
        ezmock_binary=EZMOCK_BINARY_PATH,
        **kwargs
    ):
        self.name = output_prefix
        self.params = generate_ezmock_params(output_prefix, **kwargs)
        
        if does_output_prefix_exist(output_prefix):
            self._import_ezmock_output(self.params['compute_CF'], self.params['compute_CF_zdist'])
            return None

        # timestamp we use for all generated files
        # not unique because of daylight savings time, hmmm
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # use this filename to guarantee uniqueness (jobs might be submitted within a second!)
        temp_filename = '{}-{}'.format(timestamp, output_prefix)

        # make params file for EZmock
        params_filename = '{}.ini'.format(temp_filename)
        params_file_path = os.path.join(EZMOCK_TEMP_DIR, params_filename)
        generate_ezmock_input_file(self.params, params_file_path)
        if verbose:
            print('Generated params file at {}'.format(params_file_path))
            
        # create shell script for EZmock
        sbatch_filename = '{}.sbatch'.format(temp_filename)
        sbatch_path = os.path.join(EZMOCK_TEMP_DIR, sbatch_filename)
        
        # we needed ~12m for a 5.12m catalog on 2 Gpc/h box
        # bottlenecked by 2PCF computation, which scales:
        # - linearly with volume (cube with boxsize)
        # - quadratic with effective density (density * dilute_factor)
        # so equivalently, scales as dilute^2 * N^2 / L^3
        number_ratio = self.params['expect_sum_pdf']/5120000
        side_length_ratio = self.params['boxsize'] / 2000
        time_limit_2pcf = datetime.timedelta(minutes=20) * self.params['dilute_factor']**2 * (number_ratio)**2 / (side_length_ratio)**(-3)
        MIN_TIME = datetime.timedelta(minutes=20)
        time_limit = max(MIN_TIME, time_limit_2pcf)
        self._generate_ezmock_sbatch(
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
        self.process = subprocess.run(ezmock_cmd, stdout=process_stdout)

        if self.process.returncode != 0:
            raise RuntimeError('EZmock returned with nonzero exit code')

        # copy the params file into the output directory
        # eventually make this part of fortran code
        output_params_file_name = output_prefix + '.ini'
        output_params_file_path = os.path.join(EZMOCK_OUT_DIR, output_params_file_name)
        shutil.copyfile(params_file_path, output_params_file_path)
        
        if not sbatch:
            self._import_ezmock_output(self.params['compute_CF'], self.params['compute_CF_zdist'])

        
    def _import_ezmock_output(self, import_cf_real, import_cf_zdist):
        """
        """
        self.catalog = import_ezmock_catalog(self.name, BoxSize=[self.params['boxsize']]*3)
        
        # TODO refactor this
        
        pk_path = os.path.join(EZMOCK_OUT_DIR, '{}.dat.CICassign.NGPbin.pk.mono'.format(self.name))
        pk_real = np.genfromtxt(pk_path, names=['k', 'power'])
        
        pk_path = os.path.join(EZMOCK_OUT_DIR, '{}.dat.CICassign.NGPbin.pk.zdist.mono'.format(self.name))
        pk_zdist_mono = np.genfromtxt(pk_path, names=['k', 'power'])
        
        pk_path = os.path.join(EZMOCK_OUT_DIR, '{}.dat.CICassign.NGPbin.pk.zdist.quad'.format(self.name))
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
        realmesh = self.catalog.to_mesh(
            Nmesh=nmesh,
            compensated=True,
            interlaced=True,
            position='Position',
            resampler='cic',
        )
        pk_real = nbodykit.algorithms.FFTPower(realmesh, mode='1d', dk=0.005, kmin=0).power
        self.pk = {
            'real': pk_real,
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
      
    
    @classmethod
    def _generate_ezmock_sbatch(
        cls,
        sbatch_path,
        params_file_path,
        time_limit,
        ezmock_binary='/home/users/txz/EZmock_eBOSS_LRG_ELG_empty/fortran/pro_EZmock_pk_cf',
    ):
        """
        TODO
        """
        DAY = datetime.timedelta(days=1)
        SECOND = datetime.timedelta(seconds=1)
        days, remainder = divmod(time_limit, DAY)
        total_seconds = remainder // SECOND
        rounded_remainder = datetime.timedelta(seconds=total_seconds)
        
        if days == 0:
            time_limit_str = str(rounded_remainder)
        else:
            time_limit_str = '{}-{}'.format(days, rounded_remainder)
            
#         time_limit_str = if time_limit >= DAY else str(time_limit)
        
        print('Time limit {}'.format(time_limit_str))
    
        generated_sbatch = cls.ezmock_submit_template.render(
            job_name='ezmock',
            time_limit=time_limit_str,
            params_file_path=params_file_path,
            ezmock_binary=ezmock_binary,
        )
        with open(sbatch_path, 'w+') as fp:
            fp.write(generated_sbatch)
        os.chmod(sbatch_path, 0o755)
