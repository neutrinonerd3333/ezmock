import itertools
# import json
# import os.path

import numbers
import numpy as np
import matplotlib
import matplotlib.cm as cm
# import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# import ezmock
# from ezmock import EZmock

# import nbodykit.algorithms
# import nbodykit.source.catalog


def decimal_to_filename_friendly(x, decimal_places):
    '''
    Converts numbers into filename-friendly strings. Decimal points are
    replaced by 'p', and negative signs by 'n'.
    
    Parameters
    ----------
    x : real number
        Number to convert.
    decimal_places : int
        Number of decimal places to represent.

    Returns
    -------
    str
        Filename-friendly string representation of `x`.
    '''
    num_str = f'{{:.{decimal_places}f}}'.format(x)
    return num_str.replace('.', 'p').replace('-', 'n')


def d2ff(x, places):
    '''
    Alias for `decimal_to_filename_friendly`.
    
    See Also
    --------
    decimal_to_filename_friendly
    '''
    return decimal_to_filename_friendly(x, places)


def plot_ezmock_pks(ax, ezmocks, labels, colors=None, **kwargs):
    '''
    Plot power spectra for given EZmocks.
    
    ax : matplotlib.axes.Axes
    ezmocks : list of EZmock
        EZmocks to plot PKs for.
    labels : list of str
        Labels to use to identify the individual PKs.
    colors : list of color, optional
        Color specifiers for the lines to be used.
    '''
    if colors is None:
        colors = itertools.repeat(None)
    
    for ezmock, label, color in zip(ezmocks, labels, colors):
        pk_obj = ezmock.pk['real']
        plot_pk(ax, pk_obj, label=label, color=color, linewidth=0.75, **kwargs)
    


# package UNIT into a nice object with the same interface as EZmock
# TODO make them inherit the same thing
class EnhancedCatalog():
    def __init__(self, name, catalog, pks, pcfs, bispecs):
        self.catalog = catalog
        self.pk = pks
        self.twop_cf = pcfs
        self.name = name
        self.bispec = bispecs


def compute_pks(catalog):
    mesh_kwargs = dict(
        Nmesh=256,
        compensated=True,
        interlaced=True,
        resampler='cic',
    )
    realmesh = catalog.to_mesh(
        position='Position',
        **mesh_kwargs,
    )
    realpk = nbodykit.algorithms.FFTPower(
        realmesh,
        mode='1d',
        dk=0.005,
        kmin=0,
    )
    
    zmesh = catalog.to_mesh(
        position='RSDPosition',
        **mesh_kwargs,
    )
    zpk = nbodykit.algorithms.FFTPower(
        zmesh,
        mode='2d',
        dk=0.005,
        kmin=0,
        Nmu=5,
        los=[0,0,1],
        poles=[0,2],
    )
    
    pks = { 'real': realpk, 'zdist': zpk }
    return pks


# def save_realspace_catalog(catalog, fname):
#     positions = catalog['Position'].compute()
#     np.savetxt(fname, positions, fmt='%1.3f')


# def save_zspace_catalog(catalog, fname):
#     rsd_positions = catalog['RSDPosition'].compute()

#     # nbodykit does not automagically apply periodic boundary conditions!
#     np.mod(rsd_positions, catalog.attrs['BoxSize'], out=rsd_positions)

#     np.savetxt(fname, rsd_positions, fmt='%1.3f')
    
# def import_pk_file(path):
#     return np.genfromtxt(path, names=['k', 'power'])
    
# def import_cf_file(path):
#     return np.genfromtxt(path, names=['r', 'corr'])


# def import_catalog_with_manifest(root_dir, manifest):
#     if not {'x', 'y', 'z'} <= set(manifest['columns']):
#         raise ValueError('x, y, z not among columns')
    
#     catalog = nbodykit.source.catalog.file.CSVCatalog(
#         os.path.join(root_dir, manifest['filename']),
#         manifest['columns'],
#     )
    
#     catalog['Position'] = nbodykit.transform.StackColumns(
#         catalog['x'], catalog['y'], catalog['z'],
#     )
#     if 'vx' in manifest['columns']:
#         catalog['Velocity'] = nbodykit.transform.StackColumns(
#             catalog['vx'], catalog['vy'], catalog['vz'],
#         )
#         catalog['VelocityOffset'] = manifest['rsd_factor'] * catalog['Velocity']
#         line_of_sight = [0, 0, 1]
#         catalog['RSDPosition'] = \
#             catalog['Position'] + catalog['VelocityOffset'] * line_of_sight
#     elif 'z_rsd' in manifest['columns']:
#         catalog['RSDPosition'] = nbodykit.transform.StackColumns(
#             catalog['x'], catalog['y'], catalog['z_rsd'],
#         )
#     else:
#         raise NotImplementedError()
        
#     catalog.attrs['BoxSize'] = manifest['box_size']
    
#     return catalog


        
# def import_manifested_catalog(catalog_dir_abspath):
#     root_dir = catalog_dir_abspath
    
#     manifest_path = os.path.join(root_dir, 'manifest.json')
#     with open(manifest_path, 'r') as fp:
#         manifest = json.load(fp)
    
#     catalog = None
#     if manifest['filename'] is not None:
#         catalog = import_catalog_with_manifest(root_dir, manifest)
    
#     if manifest['power_spectra']:
#         pks_dir = os.path.join(root_dir, 'pk')
#         pks = {
#             'real' : import_pk_file(os.path.join(pks_dir, 'real-mono.dat')),
#             'zdist': {
#                 'mono': import_pk_file(os.path.join(pks_dir, 'zdist-mono.dat')),
#                 'quad': import_pk_file(os.path.join(pks_dir, 'zdist-quad.dat')),
#             },
#         }
#     elif catalog is not None:
#         pks = compute_pks(catalog)
#     else:
#         raise ValueError('Cannot have no catalog and no PK!')
    
#     cfs = None
#     if manifest['corr_funcs']:
#         cfs_dir = os.path.join(root_dir, 'cf')
#         cfs = {
#             'real' : import_cf_file(os.path.join(cfs_dir, 'real-mono.dat')),
#             'zdist': {
#                 'mono': import_cf_file(os.path.join(cfs_dir, 'zdist-mono.dat')),
#                 'quad': import_cf_file(os.path.join(cfs_dir, 'zdist-quad.dat')),
#             },
#         }

#     bks = None
#     if manifest['bispectra']:
#         bks_dir = os.path.join(root_dir, 'bk')
#         bks = {
#             'real': np.genfromtxt(
#                 os.path.join(bks_dir, 'real-0p1-0p2.dat'),
#                 names=True,
#             ),
#             'zdist': np.genfromtxt(
#                 os.path.join(bks_dir, 'zdist-0p1-0p2.dat'),
#                 names=True,
#             ),
#         }
    
#     return EnhancedCatalog(manifest['short_name'], catalog, pks, cfs, bks)

    
def plot_with_xpower(ax, x, y, exponent, yerr=None, fill=False, **kwargs):
    '''
    Plot y * x**exponent against x.
    '''
    if yerr is None:
        return ax.plot(x, y * x**exponent, **kwargs)[0]
    else:
        return ax.errorbar(x, y * x**exponent, yerr * x**exponent, **kwargs)


# def plot_with_err_and_xpower(ax, x, y, yerr, exponent, fill=False, errorbars=True, fill_kw={}, **kwargs):
#     if not (errorbars or fill):
#         raise ValueError('Must have either errorbars or fill to indicate error!')

#     if errorbars:
#         errorbar = ax.errorbar(x, y * x**exponent, yerr * x**exponent, **kwargs)
#         line = errorbar[0]
#     else:
#         line, = ax.plot(x, y * x**exponent, **kwargs)
    
#     if fill:
#         ax.fill_between(x, (y + yerr) * x**exponent, (y - yerr) * x**exponent, color=line.get_color(), linewidth=0, **fill_kw)
        
#     if errorbars:
#         return errorbar
#     else:
#         return line


# def plot_2pt_stats(
#     mocks,
#     labels,
#     title=None,
# ):
#     fig, axss = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    
#     for space, axs in zip(['real', 'zdist'], axss):
#         ax1, ax2 = axs
        
#         for mock, label in zip(mocks, labels):            
#             pk = mock.pk[space].power
#             wavenums = pk['k']
#             noiseless_power = pk['power'].real - pk.attrs['shotnoise']
#             plot_with_xpower(ax1, wavenums, noiseless_power, 1.5, label=label, linewidth=0.75)

#             cf = mock.twop_cf[space]
#             plot_with_xpower(ax2, cf['r'], cf['corr'], 2, label=label, linewidth=0.75)
            
#         ax1.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
#         ax1.set_ylabel(r"$k^{1.5} P(k)$ [$(h^{-1}\mathrm{Mpc})^{1.5}$]")

#         ax2.set_xlabel(r'$r$ [Mpc/$h$]')
#         ax2.set_ylabel(r"$r^2 \xi_0$ [$(\mathrm{Mpc}/h)^2$]")
    
#     axss[0][0].legend()
    
#     if title is not None:
#         fig.suptitle(title)
        
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
    
#     return fig


# def plot_real_2pt_stats(
#     mocks,
#     labels,
#     title=None,
# ):
#     fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    
#     for mock, label in zip(mocks, labels):
#         pk = mock.pk['real'].power
#         wavenums = pk['k']
#         noiseless_power = pk['power'].real - pk.attrs['shotnoise']
#         plot_with_xpower(ax1, wavenums, noiseless_power, 1.5, label=label, linewidth=0.75)
        
#         cf = mock.twop_cf['real']
#         plot_with_xpower(ax2, cf['r'], cf['corr'], 2, label=label, linewidth=0.75)
    
    
#     ax1.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
#     ax1.set_ylabel(r"$k^{1.5} P(k)$ [$(h^{-1}\mathrm{Mpc})^{1.5}$]")
    
#     ax2.set_xlabel(r'$r$ [Mpc/$h$]')
#     ax2.set_ylabel(r"$r^2 \xi_0$ [$(\mathrm{Mpc}/h)^2$]")
    
#     ax1.legend()
    
#     if title is not None:
#         fig.suptitle(title)
        
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
    
#     return fig


def get_plot_colors(val_container, cmap=cm.cividis, cmap_bounds=None):
    if cmap_bounds is None:
        cmap_bounds = min(val_container), max(val_container)
    vmin, vmax = cmap_bounds

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return [color_mapper.to_rgba(s) for s in val_container]


# def compare_ezmocks_realspace(
#     ezmock_dict,
#     **kwargs,
# ):
#     '''
#     ezmock_dict : dict
#         dict of dicts of the form
#         {
#             a_val_0: {
#                 s_val_00: mock_00,
#                 s_val_01: mock_01,
#                 ...
#             },
#             ...
#         }
#     '''
#     ncols = len(ezmock_dict)
    
#     figsize = (4 * ncols, 3 * 2)
#     fig, axs = plt.subplots(nrows=2, ncols=ncols, sharey='row', figsize=figsize)
#     pk_axs, cf_axs = axs
    
#     is_first = True
#     for pk_ax, cf_ax, (a_val, ezmock_subdict) in zip(pk_axs, cf_axs, ezmock_dict.items()):
#         svals = ezmock_subdict.keys()
#         mocks = ezmock_subdict.values()
        
#         plot_colors = get_plot_colors(svals)
        
#         plot_pks_with_fiducial(pk_ax, mocks, svals, None, colors=plot_colors, **kwargs)
#         plot_2pcfs_with_fiducial(cf_ax, mocks, svals, None, colors=plot_colors, ylabel=is_first, legend=False, **kwargs)
        
#         pk_ax.set_title(f'expect_A_pdf = {a_val}')
#         is_first = False
    
#     for i, ax in enumerate(pk_axs):
#         setup_pk_ax(ax, ylabel=(i==0), xlabel=True, xlim=None)
#         ax.legend()
#     # pk_axs[0].legend()
    
#     fig.suptitle('EZmock PK/2pcf for various expect_A_pdf (panes), scatter2 (lines)')
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
    
#     return (fig, axs)

def compare_ezmocks_with_fiducial(
    ezmock_dict,
    scatter2s,
    expect_a_pdfs,
    fiducial=None,
    figsize=None,
    scatter_major=True
):
    major_list, minor_list = (scatter2s, expect_a_pdfs) if scatter_major else (expect_a_pdfs, scatter2s)
    major_name = 'scatter2' if scatter_major else 'expect_A_pdf'
    col_num = len(major_list)

    plot_colors = get_plot_colors(minor_list)
    
    def iterate_over_minor(major_val):
        if scatter_major:
            return [(major_val, e) for e in minor_list]
        else:
            return [(e, major_val) for e in minor_list]

    if figsize is None:
        figsize = (4 * col_num, 3 * 2)
    fig, axs = plt.subplots(
        nrows=2, ncols=col_num, sharey='row', figsize=figsize, squeeze=False,
        constrained_layout=True,
    )
    pk_axs, cf_axs = axs
    
    is_first = True
    for (pk_ax, cf_ax, major_val) in zip(pk_axs, cf_axs, major_list):
        ezmocks = [ezmock_dict[k] for k in iterate_over_minor(major_val)]
        plot_pks_with_fiducial(pk_ax, ezmocks, minor_list, fiducial, colors=plot_colors)
        plot_2pcfs_with_fiducial(cf_ax, ezmocks, minor_list, fiducial, colors=plot_colors, ylabel=is_first, legend=False)
        
        pk_ax.set_title('{} = {}'.format(major_name, major_val))
        is_first = False
    
    for i, ax in enumerate(pk_axs):
        setup_pk_ax(ax, ylabel=(i==0), xlabel=True, xlim=None)
    pk_axs[0].legend()
    
    if scatter_major:
        fig.suptitle('EZmock PK/2pcf for various scatter2 (panes), expect_A_pdf (lines)')
    else:
        fig.suptitle('EZmock PK/2pcf for various expect_A_pdf (panes), scatter2 (lines)')
    # fig.tight_layout(rect=(0, 0, 1, 0.95))
    
    return (fig, axs)


def plot_with_yscale(ax, x, y, yscale, **kwargs):
    '''
    Plot data `(x, y)` on an Axes object `ax`, with the y-axis scaled as
    specified by `yscale`.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to plot data on.
    x, y : array-like or scalar
        Same as in `matplotlib.Axes.plot`
    yscale : number or 'log'
        If 'log', plot on a log scale. If a real number, plot y * x**yscale.
    
    Returns
    -------
    lines
        A list of Line2D objects representing the plotted data.
    
    Other parameters
    ----------------
    **kwargs
        All kwargs supported by `matplotlib.Axes.plot`.
    '''
    if isinstance(yscale, str):
        if yscale == 'log':
            return ax.semilogy(x, y, **kwargs)
        raise ValueError('Only valid string value for `yscale` is "log"')
    elif not isinstance(yscale, numbers.Real):
        raise TypeError('Did not supply `log` or real value for `yscale`!')

    return plot_with_xpower(ax, x, y, yscale, **kwargs)


def plot_pk(ax, pk_object, yscale=1.5, start_ind=1, **kwargs):
    Pk = pk_object.power
    wavenums = Pk['k']
    noiseless_power = Pk['power'].real - Pk.attrs['shotnoise']
    
    plot_with_yscale(ax, wavenums[start_ind:], noiseless_power[start_ind:], yscale, **kwargs)

    
# def plot_pk_pole(ax, pk_object, pole, yscale=1.5, **kwargs):
#     if pole not in [0, 2]:
#         raise ValueError(f'pole must be 0 or 2, was {pole}')
#     poles = pk_object.poles
    
#     k = poles['k']
#     power = poles[f'power_{pole}'].real - (pk_object.attrs['shotnoise'] if pole == 0 else 0)
    
#     plot_with_yscale(ax, k, power, yscale, **kwargs)
    
    
# def plot_denoised_pk_real(ax, pk_arr, **kwargs):
#     plot_with_xpower(ax, pk_arr['k'], pk_arr['power'], 1.5, **kwargs)

# def plot_denoised_pk_zdist0(ax, pk_arr, **kwargs):
#     plot_with_xpower(ax, pk_arr['k'], pk_arr['power_0'], 1.5, **kwargs)

# def plot_denoised_pk_zdist2(ax, pk_arr, **kwargs):
#     plot_with_xpower(ax, pk_arr['k'], pk_arr['power_2'], 1.5, **kwargs)

    
# def plot_cf(ax, cf, **kwargs):
#     plot_with_xpower(ax, cf['r'], cf['corr'], 2, **kwargs)


def plot_pks_with_fiducial(ax, ezmocks, labels, fiducial, colors=None, **kwargs):
    if fiducial is not None:
        plot_pk(ax, fiducial.pk['real'], label=fiducial.name, color='r', linewidth=2)
    plot_ezmock_pks(ax, ezmocks, labels, colors=colors, **kwargs)


def plot_2pcfs_with_fiducial(ax, ezmocks, labels, fiducial=None, colors=None, ylabel=True, legend=True, **kwargs):
    if fiducial is not None:
        fid_2pcf = fiducial.twop_cf['real']
        plot_with_xpower(ax, fid_2pcf['r'], fid_2pcf['corr'], 2, label=fiducial.name, color='r', linewidth=2)
    
    if colors is None:
        colors = itertools.repeat(None)
    
    for ezmock, label, color in zip(ezmocks, labels, colors):
        cf = ezmock.twop_cf['real']
        plot_with_xpower(ax, cf['r'], cf['corr'], 2, label=label, color=color, linewidth=0.75, **kwargs)
    
    ax.set_xlabel(r'$r$ [Mpc/$h$]')
    if ylabel:
        ax.set_ylabel(r"$r^2 \xi_0$ [$(\mathrm{Mpc}/h)^2$]")
    if legend:
        ax.legend()


def plot_bk(ax, bispec_arr, **kwargs):
    thetas = bispec_arr['theta']
    bs = bispec_arr['B']
    ax.plot(thetas / np.pi, bs, **kwargs)


# def plot_mock_bks(axs, mock, **kwargs):
#     plot_bispec(axs[0], mock.bispec['real'], linewidth=1, **kwargs)
#     plot_bispec(axs[1], mock.bispec['zdist'], linewidth=1, **kwargs)
    

# def setup_bk_fig(fig, axs):
#     setup_bk_ax(axs[0], ylabel=False)
#     setup_bk_ax(axs[1], ylabel=False)
    
#     fig.suptitle(r'$B(\theta)$ ($k_2 = 2 k_1 = 0.2 h/\mathrm{Mpc}$)')
#     axs[0].set_ylabel('real space')
#     axs[1].set_ylabel('$z$ space')

    
def mocks_bispec_comparison(ezmocks, colors, labels, fiducial, fid_label, legend=True):
    fig, [ax1, ax2] = plt.subplots(
        nrows=2,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={ 'height_ratios': [2, 1] },
    )
    
    catcollabs = zip(ezmocks, colors, labels)
    catcollabs = itertools.chain(catcollabs, [[fiducial, 'red', fid_label]])
    
    for catalog, color, label in catcollabs:
        plot_bk(ax1, catalog.bispec['real'], linewidth=1, label=f'{label} real', color=color)
        plot_bk(ax1, catalog.bispec['zdist'], linewidth=1, linestyle='dashed', label=f'{label} zdist', color=color)
    
    cat_col_lab_zip = zip(ezmocks, colors, labels)
    for catalog, color, label in cat_col_lab_zip:
        thetas = catalog.bispec['real']['theta']
        ratio_real = catalog.bispec['real']['B'] / fiducial.bispec['real']['B']
        ax2.plot(thetas / np.pi, ratio_real, color=color, label=f'{label} real')
    
        ratio_zdist = catalog.bispec['zdist']['B'] / fiducial.bispec['zdist']['B']
        ax2.plot(thetas / np.pi, ratio_zdist, linestyle='dashed', color=color, label=f'{label} zdist')

    ax2.set_ylabel('Ratio (E/U)')
    ax2.set_xlabel(r'$\theta/\pi$')
    ax2.axhline(1, linestyle='dashed', color='0.5')
    ax1.set_ylabel(r'$B(\theta)$')
    
    if legend:
        ax1.legend()
        
    ax1.set_title('Bispectrum for $k_2 = 2k_1 = 0.2 h/\mathrm{Mpc}$')
    
    return fig, [ax1, ax2]

    
def fog_fitting(ezmocks, fiducial=None):
    fogs = list(ezmocks.keys())
    
    fog_colors = get_plot_colors(fogs)
    
    labels = fogs
    colors = fog_colors
    catalogs = list(ezmocks.values())
    linewidths = [0.5] * len(fogs)
    if fiducial is not None:
        labels += [fiducial.name]
        colors += ['red']
        catalogs += [fiducial]
        linewidths += [1]

    return plot_mock_2pts(catalogs, labels, colors, linewidths)


# def make_2pt_clustering_axs():
#     fig = plt.figure(constrained_layout=True, figsize=(12,8))
#     gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[2,1,2,1])
    
#     def create_plot_row(row_ind, sharex_axs=None):
#         if sharex_axs is None:
#             sharex_axs = [None] * 4

#         first = fig.add_subplot(gs[row_ind, 0], sharex=sharex_axs[0])
#         axs = [first]
#         for i in range(1, 3):
#             axs.append(fig.add_subplot(
#                 gs[row_ind, i], sharex=sharex_axs[i], sharey=first
#             ))
#         return axs
    
#     pk_axs = create_plot_row(0)
#     pk_ratio_axs = create_plot_row(1, sharex_axs=pk_axs)
#     cf_axs = create_plot_row(2)
#     cf_ratio_axs = create_plot_row(3, sharex_axs=cf_axs)
    
#     data_axs = [pk_axs, cf_axs]
#     ratio_axs = [pk_ratio_axs, cf_ratio_axs]
    
#     setup_pk_ax(pk_axs[0], xlabel=False, ylabel=True)
#     setup_pk_ax(pk_axs[1], pole=0, xlabel=False, ylabel=True)
#     setup_pk_ax(pk_axs[2], pole=2, xlabel=False, ylabel=True)
#     setup_cf_ax(cf_axs[0], xlabel=False, ylabel=True)
#     setup_cf_ax(cf_axs[1], pole=0, xlabel=False, ylabel=True)
#     setup_cf_ax(cf_axs[2], pole=2, xlabel=False, ylabel=True)
    
#     for ax in pk_ratio_axs:
#         setup_pk_ax(ax, xlabel=True, ylabel=False)
#     for ax in cf_ratio_axs:
#         setup_cf_ax(ax, xlabel=True, ylabel=False)
        
#     for ratio_ax_row in ratio_axs:
#         ratio_ax_row[0].set_ylabel('ratio')
#         for ax in ratio_ax_row:
#             setup_ratio_ax(ax)
    
#     axs = np.array([pk_axs, pk_ratio_axs, cf_axs, cf_ratio_axs])
    
#     # turn off extra tick labels
#     for ax in axs[:,1:].flatten():
#         plt.setp(ax.get_yticklabels(), visible=False)
#     for ax in pk_axs:
#         plt.setp(ax.get_xticklabels(), visible=False)
#     for ax in cf_axs:
#         plt.setp(ax.get_xticklabels(), visible=False)
    
    
#     return (fig, axs)


# def plot_mock_2pts_with_fiducial(mocks, fiducial, colors, labels, fiducial_name, legend=True):
#     fig, axs = make_2pt_clustering_axs()
#     pk_axs, pk_ratio_axs, cf_axs, cf_ratio_axs = axs

#     data_axs = [pk_axs, cf_axs]
#     ratio_axs = [pk_ratio_axs, cf_ratio_axs]
    
#     for (mock, color, label) in zip(mocks, colors, labels):
#         plot_single_2pt_stats(
#             data_axs,
#             mock,
#             linewidth=0.8,
#             color=color,
#             label=label,
#         )
#         plot_2pt_fiducial_ratios(
#             ratio_axs,
#             mock,
#             fiducial,
#             color=color,
#             label=label,
#         )
        
#     plot_single_2pt_stats(
#         data_axs,
#         fiducial,
#         linewidth=1.2,
#         color='red',
#         label=fiducial_name,
#     )
    
    
#     if legend:
#         pk_axs[0].legend()
    

def plot_single_2pt_stats(axs, catalog, **kwargs):
    pkaxs, cfaxs = axs
    
    realpk = catalog.pk['real'].power
    plot_with_xpower(
        pkaxs[0],
        realpk['k'],
        realpk['power'].real - realpk.attrs['shotnoise'],
        1.5,
        **kwargs,
    )
    
    zpk = catalog.pk['zdist'].poles
    plot_with_xpower(pkaxs[1], zpk['k'], zpk['power_0'].real - zpk.attrs['shotnoise'], 1.5, **kwargs)
    plot_with_xpower(pkaxs[2], zpk['k'], zpk['power_2'].real, 1.5, **kwargs)


    cfs = catalog.twop_cf
    cfs_real = cfs['real']
    cfs_mono = cfs['zdist']['mono']
    cfs_quad = cfs['zdist']['quad']

    plot_with_xpower(
        cfaxs[0],
        cfs_real['r'],
        cfs_real['corr'],
        2,
        **kwargs,
    )
    plot_with_xpower(cfaxs[1], cfs_mono['r'], cfs_mono['corr'], 2, **kwargs)
    plot_with_xpower(cfaxs[2], cfs_quad['r'], cfs_quad['corr'], 2, **kwargs)

    
# def extract_2pt_stats(catalog):
#     realpk = catalog.pk['real'].power
#     k, pk = realpk['k'], realpk['power'].real - realpk.attrs['shotnoise']
    
#     zpk = catalog.pk['zdist'].poles
#     zk, zpk0, zpk2 = zpk['k'], zpk['power_0'].real - zpk.attrs['shotnoise'], zpk['power_2'].real
    
#     fourier_stats = (k, pk, zk, zpk0, zpk2)
    
#     cfs = catalog.twop_cf
#     cfs_real = cfs['real']
#     cfs_mono = cfs['zdist']['mono']
#     cfs_quad = cfs['zdist']['quad']
#     real_stats = (cfs_real, cfs_mono, cfs_quad)
    
#     return (fourier_stats, real_stats)


# def plot_2pt_fiducial_ratios(axs, catalog, fiducial, **kwargs):
#     pkaxs, cfaxs = axs
    
#     fourier_stats, real_stats = extract_2pt_stats(catalog)
#     k, pk, zk, zpk0, zpk2 = fourier_stats
#     cfs_real, cfs_mono, cfs_quad = real_stats
    
#     fourier_stats_f, real_stats_f = extract_2pt_stats(fiducial)
#     kf, pkf, zkf, zpk0f, zpk2f = fourier_stats_f
#     cfs_real_f, cfs_mono_f, cfs_quad_f = real_stats_f
    
#     pkaxs[0].plot(k, pk/pkf, **kwargs)
#     pkaxs[1].plot(zk, zpk0/zpk0f, **kwargs)
#     pkaxs[2].plot(zk, zpk2/zpk2f, **kwargs)
    
#     cfaxs[0].plot(cfs_real['r'], cfs_real['corr']/cfs_real_f['corr'], **kwargs)
#     cfaxs[1].plot(cfs_mono['r'], cfs_mono['corr']/cfs_mono_f['corr'], **kwargs)
#     cfaxs[2].plot(cfs_quad['r'], cfs_quad['corr']/cfs_quad_f['corr'], **kwargs)
    

def setup_pk_ax(ax, pole=None, xlabel=False, ylabel=False, xlim=None):
    if xlim is None:
        pass
    elif xlim is True:
        ax.set_xlim([-0.02,0.42])
    else:
        ax.set_xlim(xlim)
    
    if xlabel:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
    if ylabel:
        if pole is None:
            ax.set_ylabel(r'$k^{1.5} P(k)$ [$(\mathrm{Mpc}/h)^{1.5}$]')
        elif pole == 0:
            ax.set_ylabel(r'$k^{1.5} P_0(k)$ [$(\mathrm{Mpc}/h)^{1.5}$]')
        elif pole == 2:
            ax.set_ylabel(r'$k^{1.5} P_2(k)$ [$(\mathrm{Mpc}/h)^{1.5}$]')


def setup_cf_ax(ax, pole=None, xlabel=False, ylabel=False):
    if xlabel:
        ax.set_xlabel(r'$r$ [Mpc/$h$]')
    if ylabel:
        if pole is None:
            ax.set_ylabel(r'$r^2 \xi(r)$ [$(\mathrm{Mpc}/h)^2$]')
        elif pole == 0:
            ax.set_ylabel(r'$r^2 \xi_0(r)$ [$(\mathrm{Mpc}/h)^2$]')
        elif pole == 2:
            ax.set_ylabel(r'$r^2 \xi_2(r)$ [$(\mathrm{Mpc}/h)^2$]')
    

def setup_bk_ax(ax, xlabel=True, ylabel=True):
    if xlabel:
        ax.set_xlabel(r'$\theta/\pi$')
    if ylabel:
        ax.set_ylabel(r'$B(\theta)$ ($k_2 = 2 k_1 = 0.2 h/\mathrm{Mpc}$)')


def setup_2pt_stats_axs(axs):
    [pkaxs, cfaxs] = axs
    
    for ax in pkaxs:
        setup_pk_ax(ax, xlabel=True)
    for ax in cfaxs:
        setup_cf_ax(ax, xlabel=True)
    
    pkaxs[0].set_ylabel(r'$k^{1.5} P(k)$ [$(\mathrm{Mpc} / h)^{1.5}$]')
    pkaxs[1].set_ylabel(r'$k^{1.5} P_0(k)$ [$(\mathrm{Mpc} / h)^{1.5}$]')
    pkaxs[2].set_ylabel(r'$k^{1.5} P_2(k)$ [$(\mathrm{Mpc} / h)^{1.5}$]')

    cfaxs[0].set_ylabel(r'$r^2 \xi(r)$ [$(\mathrm{Mpc}/h)^2$]')
    cfaxs[1].set_ylabel(r'$r^2 \xi_0(r)$ [$(\mathrm{Mpc}/h)^2$]')
    cfaxs[2].set_ylabel(r'$r^2 \xi_2(r)$ [$(\mathrm{Mpc}/h)^2$]')


# def setup_ratio_ax(ax, **kwargs):
#     # ax.set_ylim(0.7, 1.3)
#     default_kwargs = dict(color='0.5', linestyle='dashed', linewidth=0.5)
#     kwargs = dict(default_kwargs, **kwargs)
#     ax.axhline(1, **kwargs)
    

def plot_mock_2pts(catalogs, labels, colors, linewidths, title=None):
    fig, axs = plt.subplots(
        nrows=2, ncols=3,
        figsize=(12, 6),
        constrained_layout=True,
        sharey='row',
    )
    [pkaxs, cfaxs] = axs
    
    for label, color, catalog, lw in zip(labels, colors, catalogs, linewidths):
        kwargs = {
            'label': str(label),
            'color': color,
            'linewidth': lw,
        }
        plot_single_2pt_stats(axs, catalog, **kwargs)
    
    pkaxs[0].legend()
    
    for ax in cfaxs:
        ax.set_xlabel(r'$r$ [Mpc/$h$]')
    for ax in pkaxs:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')

    setup_2pt_stats_axs(axs)

    if title is not None:
        fig.suptitle(title)
    
    
    return fig, axs
