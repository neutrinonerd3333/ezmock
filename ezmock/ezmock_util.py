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


class EZmock():
    """
    A Python wrapper around individual EZmock instances.

    Parameters
    ----------
    """

    def __init__(self, output_prefix, output_dir=EZMOCK_OUT_DIR):
        self.name = output_prefix
        self.output_dir = output_dir
        with open(os.path.join(output_dir, output_prefix + '.pickle'), 'rb') as fp:
            self.params = pickle.load(fp)
        self._import_ezmock_output(self.params['compute_CF'], self.params['compute_CF_zdist'])
        self._bispec = None


    @property
    def bispec(self):
        if self._bispec is not None:
            return self._bispec

        realspace_output_file = os.path.join(self.output_dir, f'{self.name}.dat.CICassign.bispec')
        zspace_output_file = os.path.join(self.output_dir, f'{self.name}.dat.CICassign.bispec.zdist')

        # TODO make this configurable!
        bispec_binary = os.path.join(os.environ['HOME'], 'codez', 'bispec_box', 'bispec')
        conf_file = os.path.join(os.environ['HOME'], 'codez', 'bispec_box', 'unit-fits-bispec.conf')
        env = dict(os.environ, OMP_NUM_THREADS='4')

        boxsize = self.params['boxsize']

        if not os.path.isfile(realspace_output_file):
            catalog_file = self._catalog_fname()
            bispec_cmd = f'{bispec_binary} --conf={conf_file} --box-max={boxsize} --input={catalog_file} --output={realspace_output_file}'
            cmd = f'module load gsl && {bispec_cmd}'
            subprocess.run(
                cmd,
                env=env,
                check=True,
                shell=True,
            )

        if not os.path.isfile(zspace_output_file):
            self._save_zspace_catalog()

            zspace_catalog_file = self._zspace_catalog_fname()
            bispec_cmd = f'{bispec_binary} --conf={conf_file} --box-max={boxsize} --input={zspace_catalog_file} --output={zspace_output_file}'
            subprocess.check_output(
                f'module load gsl && {bispec_cmd}',
                env=env,
                shell=True
            )

        self._bispec = {
            'real': np.genfromtxt(realspace_output_file, names=True),
            'zdist': np.genfromtxt(zspace_output_file, names=True),
        }

        return self._bispec

    def _import_ezmock_output(self, import_cf_real, import_cf_zdist):
        """
        """
        self.catalog = self._import_ezmock_catalog()

        # TODO refactor this

        pk_path = os.path.join(self.output_dir, f'{self.name}.dat.CICassign.NGPbin.pk.mono')
        pk_real = np.genfromtxt(pk_path, names=['k', 'power'])

        pk_path = os.path.join(self.output_dir, f'{self.name}.dat.CICassign.NGPbin.pk.zdist.mono')
        pk_zdist_mono = np.genfromtxt(pk_path, names=['k', 'power'])

        pk_path = os.path.join(self.output_dir, f'{self.name}.dat.CICassign.NGPbin.pk.zdist.quad')
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
            cf_path = os.path.join(self.output_dir, cf_file)
            self.twop_cf['real'] = np.genfromtxt(cf_path, names=['r', 'corr'])
        if import_cf_zdist:
            self.twop_cf['zdist'] = dict()

            cf_file = '{}.dat.bin5.corr.zdist.mono'.format(self.name)
            cf_path = os.path.join(self.output_dir, cf_file)
            self.twop_cf['zdist']['mono'] = np.genfromtxt(cf_path, names=['r', 'corr'])

            cf_file = '{}.dat.bin5.corr.zdist.quad'.format(self.name)
            cf_path = os.path.join(self.output_dir, cf_file)
            self.twop_cf['zdist']['quad'] = np.genfromtxt(cf_path, names=['r', 'corr'])

    def _catalog_fname(self):
        return os.path.join(self.output_dir, f'{self.name}.dat')

    def _zspace_catalog_fname(self):
        return os.path.join(self.output_dir, f'{self.name}.dat.zdist')

    def _import_ezmock_catalog(self, **kwargs):
        """
        Import a catalog generated by EZmock with given `name_stem`
        """
        fname = self._catalog_fname()
        column_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        catalog = nbodykit.source.catalog.file.CSVCatalog(fname, column_names)

        catalog['Position'] = nbodykit.transform.StackColumns(
            catalog['x'], catalog['y'], catalog['z'],
        )
        catalog['Velocity'] = nbodykit.transform.StackColumns(
            catalog['vx'], catalog['vy'], catalog['vz'],
        )

        # see nbodykit docs for meaning
        catalog['VelocityOffset'] = self.params['rsd_factor'] * catalog['Velocity']

        line_of_sight = [0, 0, 1]
        catalog['RSDPosition'] = catalog['Position'] + catalog['VelocityOffset'] * line_of_sight

        catalog.attrs['BoxSize']= [self.params['boxsize']]*3

        catalog.los = line_of_sight
        for key, val in kwargs.items():
            catalog.attrs[key] = val

        return catalog


    def _save_zspace_catalog(self):
        fname = self._zspace_catalog_fname()

        rsd_positions = self.catalog['RSDPosition'].compute()

        # nbodykit does not automagically apply periodic boundary conditions!
        np.mod(rsd_positions, self.params['boxsize'], out=rsd_positions)

        np.savetxt(fname, rsd_positions, fmt='%1.3f')
