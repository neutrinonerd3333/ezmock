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


class EZmock():
    """
    A Python wrapper around individual EZmock instances.

    Parameters
    ----------
    """
    
#     _current_dir = os.path.dirname(os.path.realpath(__file__))
#     _template_dir = os.path.join(_current_dir, '../templates')
#     jinja_loader = jinja2.FileSystemLoader(_template_dir)  # jinja wants abs path
#     jinja_template_env = jinja2.Environment(loader=jinja_loader)
#     ezmock_submit_template = jinja_template_env.get_template('ezmock-submit-template.sbatch')
    
#     @classmethod
#     def generate(
#         cls,
#         output_prefix,
#         verbose=False,
#         sbatch=False,
#         ezmock_binary=EZMOCK_BINARY_PATH,
#         **kwargs,
#     ):
#         if does_output_prefix_exist(output_prefix):
#             raise UserWarning(f'output_prefix {output_prefix} already exists!')
#             return
        
#         params, coparams = generate_ezmock_params(output_prefix, **kwargs)
        
#         # timestamp we use for all generated files
#         # not unique because of daylight savings time, hmmm
#         timestamp = time.strftime('%Y%m%d-%H%M%S')

#         # use this filename to guarantee uniqueness (jobs might be submitted within a second!)
#         temp_filename = f'{timestamp}-{output_prefix}'

#         # make params file for EZmock
#         params_filename = f'{temp_filename}.ini'
#         params_file_path = os.path.join(EZMOCK_TEMP_DIR, params_filename)
#         generate_ezmock_input_file(params, params_file_path)
#         if verbose:
#             print(f'Generated params file at {params_file_path}')
            
#         # create shell script for EZmock
#         sbatch_filename = f'{temp_filename}.sbatch'
#         sbatch_path = os.path.join(EZMOCK_TEMP_DIR, sbatch_filename)

#         time_limit = _compute_time_limit(
#             params['boxsize'],
#             params['expect_sum_pdf'],
#             params['dilute_factor']
#         )

#         cls._generate_ezmock_sbatch(
#             sbatch_path,
#             params_file_path,
#             time_limit,
#             ezmock_binary=ezmock_binary,
#         )
#         if verbose:
#             print('Generated sbatch at {}'.format(sbatch_path))

#         # run EZmock
#         ezmock_cmd = ['sbatch', shlex.quote(sbatch_path)] if sbatch else [shlex.quote(sbatch_path)]
#         print(' '.join(ezmock_cmd))

#         # make sure subprocess stdout comes out!
#         process_stdout = None if verbose else subprocess.DEVNULL
#         process = subprocess.run(ezmock_cmd, stdout=process_stdout)

#         if process.returncode != 0:
#             raise RuntimeError('EZmock returned with nonzero exit code')

#         # copy the params file into the output directory
#         # eventually make this part of fortran code
#         output_params_file_name = output_prefix + '.ini'
#         output_params_file_path = os.path.join(EZMOCK_OUT_DIR, output_params_file_name)
#         shutil.copyfile(params_file_path, output_params_file_path)
        
#         params_pickle = os.path.join(EZMOCK_OUT_DIR, output_prefix + '.pickle')
#         with open(params_pickle, 'wb') as fp:
#             pickle.dump(coparams, fp)
    
    
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

#     @classmethod
#     def _generate_ezmock_sbatch(
#         cls,
#         sbatch_path,
#         params_file_path,
#         time_limit,
#         ezmock_binary=EZMOCK_BINARY_PATH,
#     ):
#         """
#         Parameters
#         ----------
#         sbatch_path : str
#         params_file_path : str
#         time_limit : datetime.timedelta
#             time limit for the slurm job
#         ezmock_binary : str
#             path to the EZmock binary
#         """
#         time_limit_str = cls._slurm_time_format(time_limit)
#         print('Time limit = {}'.format(time_limit_str))
    
#         generated_sbatch = cls.ezmock_submit_template.render(
#             job_name='ezmock',
#             time_limit=time_limit_str,
#             params_file_path=params_file_path,
#             ezmock_binary=ezmock_binary,
#         )
#         with open(sbatch_path, 'w+') as fp:
#             fp.write(generated_sbatch)
#         os.chmod(sbatch_path, 0o755)
        
#     @staticmethod
#     def _slurm_time_format(timedelta):
#         """
#         Format the duration `timedelta` in the Slurm time limit format,
#         as specified in the sbatch(1) man page. Rounds down to the nearest
#         second.
        
#         Parameters
#         ----------
#         timedelta : datetime.timedelta
#             Duration to format.
            
#         Returns
#         -------
#         time_str : str
#             Slurm-formatted string.
#         """
#         DAY = datetime.timedelta(days=1)
#         SECOND = datetime.timedelta(seconds=1)
#         days, remainder = divmod(timedelta, DAY)
#         total_seconds = remainder // SECOND
#         rounded_remainder = datetime.timedelta(seconds=total_seconds)
        
#         if days == 0:
#             time_str = str(rounded_remainder)
#         else:
#             time_str = '{}-{}'.format(days, rounded_remainder)

#         return time_str
        
