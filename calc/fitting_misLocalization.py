"""
This file provides classes and functions for modeling and fitting the
diffraction-limited images produced by a dipole emitter coupled to a
plasmonic nanorod, modeled as a polarizable point dipole with
polarizability of a prolate ellipsoid in the modified long wavelength
approximation.
"""
from __future__ import print_function
from __future__ import division

import pdb
import sys
import os
import yaml

import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import scipy.special as spf
from scipy import interpolate

from ..optics import diffraction_int as diffi
from ..optics import fibonacci as fib

## Read parameter file to obtain fields
from misloc_mispol_package import project_path

## plotting stuff
import matplotlib.pyplot as plt
import matplotlib as mpl
#
## colorbar stuff
from mpl_toolkits import axes_grid1
#
mpl.rcParams['text.usetex'] = True
mpl.rcParams["lines.linewidth"]

## analytic image fields
from ..optics import anal_foc_diff_fields as afi
## Coupled dipole analytics
from . import coupled_dipoles as cp
from . import knn

## Get path to directory for mispolariation mapping
txt_file_path = project_path + '/txt'

## Import physical constants
phys_const_file_name = '/physical_constants.yaml'
parameter_files_path = (
    project_path + '/param')
opened_constant_file = open(
    parameter_files_path+phys_const_file_name,
    'r')
#
constants = yaml.load(opened_constant_file, Loader=yaml.SafeLoader) # (Zu Edit: Loader=yaml.SafeLoader)
e = constants['physical_constants']['e']
c = constants['physical_constants']['c']  # charge of electron in statcoloumbs
hbar = constants['physical_constants']['hbar']
cm_per_nm = constants['physical_constants']['nm']
n_a = constants['physical_constants']['nA']   # Avogadro's number

# Draw the rod and ellipse
curly_nanorod_color = (241/255, 223/255, 182/255)
curly_nanorod_color_light = (241/255, 223/255, 182/255, 0.5)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_param_file(file_name):
    """ Load parameter .yaml"""
    ## Check/fix formatting
    if file_name[0] != '/':
        file_name = '/'+file_name
    if file_name[-5:] != '.yaml':
        # print(file_name[-5:])
        file_name = file_name+'.yaml'
    ## Load
    loaded_params = yaml.load(
        open(parameter_files_path+file_name, 'r'),
        Loader=yaml.SafeLoader) # (Zu Edit: Loader=yaml.SafeLoader)
    return loaded_params

class DipoleProperties(object):
    """ Will eventually call parameter file as argument, currently (02/07/19)
        just loads relevant values from hardcoded paths. ew.

        11/21/19: Rewriting to take parameter file as input.
        """

    def __init__(self,
        param_file=None,
        eps_inf=None,
        hbar_omega_plasma=None,
        hbar_gamma_drude=None,
        a_long_in_nm=None,
        a_short_in_nm=None,
        eps_b=None,
        fluo_ext_coef=None,
        fluo_mass_hbar_gamma=None,
        fluo_nr_hbar_gamma=None,
        fluo_quench_region_nm=0,
        isolate_mode=None,
        drive_energy_eV=None,
        omega_plasma=None,
        gamma_drude=None,
        a_long_meters=None,
        a_short_meters=None,
        fluo_quench_range=None,
        drive_amp=None,
        is_sphere=None,
        sphere_model=None,
        **kwargs
        ):

        ## If parameter file is given, load all relevent parameters.
        if param_file is not None:
            self.parameters = load_param_file(param_file)
            ## Define all parameters as instance attributes.
            self.drive_energy_eV = self.parameters['general']['drive_energy']
            self.eps_inf = self.parameters['plasmon']['fit_eps_inf']
            self.omega_plasma = (
                self.parameters['plasmon']['fit_hbar_wp'] / hbar
                )
            self.gamma_drude = (
                self.parameters['plasmon']['fit_hbar_gamma'] / hbar
                )
            self.a_long_meters = (
                self.parameters['plasmon']['fit_a1'] * cm_per_nm
                )
            self.a_short_meters = (
                self.parameters['plasmon']['fit_a2'] * cm_per_nm
                )
            ## Check if Sphere
            self.true_a_un_me = self.parameters['plasmon']['true_a_unique']
            if self.true_a_un_me is None or self.true_a_un_me == 'None':
                self.is_sphere = True
            elif self.true_a_un_me is not None or self.true_a_un_me != 'None':
                self.is_sphere = False
                self.true_a_un_me *= cm_per_nm

            self.true_a_de_me = (
                self.parameters['plasmon']['true_a_degen']*cm_per_nm)
            n_b = self.parameters['general']['background_ref_index']
            self.eps_b = n_b**2.

            try: ## Checking for instructions to only quench over the real
                ## particle.
                self.quench_over_real_disk = (
                    self.parameters['plasmon']['quench_over_real_disk']
                    )
            except: print("No 'quench_over_real_disk' param found in file.")

            try:## loading frmo parameter file, not previously implemented
                ## before 11/21/19.
                self.fluo_quench_range = (
                    self.parameters['plasmon']['quench_radius']
                    )
            except:## Load default kwarg
                self.fluo_quench_range = fluo_quench_region_nm
            ## And some that will only get used in creation of the molecule
            ## polarizabity, and therefore don't need to be instance
            ## attributes.
            self.fluo_ext_coef = (
                self.parameters['fluorophore']['extinction_coeff'])
            self.fluo_mass_hbar_gamma = (
                self.parameters['fluorophore']['mass_gamma'])
            self.fluo_nr_hbar_gamma = (
                self.parameters['fluorophore']['test_gamma'])

            self.drive_amp = self.parameters['general']['drive_amp']
        ## If no parameter file is given, assume all variables are given.
        else:## Assume all parameters given explixitly as class args.
            ## Make sure no values were missed
            relevant_attr = [
                drive_energy_eV,
                eps_inf,
                omega_plasma,
                gamma_drude,
                a_long_meters,
                a_short_meters,
                eps_b,
                fluo_quench_range
                ]
            relevant_attr_names = [
                'self.drive_energy_eV',
                'self.eps_inf',
                'self.omega_plasma',
                'self.gamma_drude',
                'self.a_long_meters',
                'self.a_short_meters',
                'self.eps_b',
                'self.fluo_quench_range'
                ]
            if None in relevant_attr:
                raise ValueError(
                    'Need all attributes with manual input.\n',
                    f'missing {relevant_attr_names[relevant_attr.index(None)]}'
                    )
            ## Otherwize sotre all those args as instance attrs
            self.drive_energy_eV = drive_energy_eV
            self.eps_inf = eps_inf
            self.omega_plasma = omega_plasma
            self.gamma_drude = gamma_drude
            self.a_long_meters = a_long_meters
            self.a_short_meters = a_short_meters
            self.eps_b = eps_b
            ## hardcoded region around nanoparticle to through out results because
            ## dipole approximation at small proximities
            self.fluo_quench_range = fluo_quench_region_nm

            self.fluo_ext_coef = fluo_ext_coef
            self.fluo_mass_hbar_gamma = fluo_mass_hbar_gamma
            self.fluo_nr_hbar_gamma = fluo_nr_hbar_gamma

            self.is_sphere = is_sphere
            self.drive_amp = drive_amp
            self.parameters = None

        self.alpha0_diag_dyad = cp.sparse_polarizability_tensor(
            ## This one is a little hacky, will need to fix for proper
            ## spectral reshaping later.
            mass=cp.fluorophore_mass(
                ext_coef=self.fluo_ext_coef, # parameters['fluorophore']['extinction_coeff'],
                gamma=self.fluo_mass_hbar_gamma/hbar, # parameters['fluorophore']['mass_gamma']/hbar
                n_b=np.sqrt(self.eps_b)
                ),
            w_res=self.drive_energy_eV/hbar,
            w=self.drive_energy_eV/hbar,
            gamma_nr=self.fluo_nr_hbar_gamma/hbar, # parameters['fluorophore']['test_gamma']/hbar,
            a=0,
            eps_inf=1,
            eps_b=1
            )

        self.sphere_model = sphere_model
        if self.is_sphere:

            if self.sphere_model == 'MLWA':
                self.alpha1_diag_dyad = (
                    cp.sparse_ret_sphere_polarizability_Drude(
                        w=self.drive_energy_eV/hbar,
                        eps_inf=self.eps_inf,
                        w_p= self.omega_plasma,
                        gamma=self.gamma_drude,
                        eps_b=self.eps_b,
                        a=self.a_long_meters,
                        isolate_mode=isolate_mode)
                    )

            elif self.sphere_model == 'TMatExp':
                self.alpha1_diag_dyad = (
                    cp.sparse_TMatExp_sphere_polarizability_Drude(
                        w=self.drive_energy_eV/hbar,
                        eps_inf=self.eps_inf,
                        w_p= self.omega_plasma,
                        gamma=self.gamma_drude,
                        eps_b=self.eps_b,
                        a=self.a_long_meters,
                        isolate_mode=isolate_mode)
                    )

            elif self.sphere_model == 'Mie':
                self.alpha1_diag_dyad = (
                    cp.sparse_Mie_sphere_polarizability_Drude(
                        w=self.drive_energy_eV/hbar,
                        eps_inf=self.eps_inf,
                        w_p= self.omega_plasma,
                        gamma=self.gamma_drude,
                        eps_b=self.eps_b,
                        a=self.a_long_meters,
                        isolate_mode=isolate_mode)
                    )


        elif not self.is_sphere:

            self.alpha1_diag_dyad = (
                cp.sparse_ret_prolate_spheroid_polarizability_Drude(
                    self.drive_energy_eV/hbar,
                    self.eps_inf,
                    self.omega_plasma,
                    self.gamma_drude,
                    self.eps_b,
                    self.a_long_meters,
                    self.a_short_meters,
                    isolate_mode=isolate_mode)
                )


class BeamSplitter(object):
    """ Class for calculating average image polarization as Curly does
        experimentally
        """

    def __init__(self,
        drive_I=None,
        sensor_size=None,
        param_file=None,
        **kwargs
        ):
        ## Truth value
        optics_params_given = None not in [drive_I, sensor_size]

        if param_file is None and optics_params_given:
            self.drive_I = drive_I
            self.sensor_size = sensor_size

        elif param_file is None and drive_I is None:
            ## Load default
            # parameters = yaml.load(
            #     open(parameter_files_path+curly_yaml_file_name, 'r')
            #     )
            raise ValueError('Default parameter file has not been implemented'+
                ' because it seems like a recipe for mistakes')
        else:
            ## Load given parameter file.
            self.parameters = load_param_file(param_file)
            self.drive_I = np.abs(self.parameters['general']['drive_amp'])**2.
            self.sensor_size = self.parameters['optics']['sensor_size']*cm_per_nm


    def powers_and_angels(self,E):
        drive_I = np.abs(self.parameters['general']['drive_amp'])**2.

        normed_Ix = np.abs(E[0])**2. / self.drive_I
        normed_Iy = np.abs(E[1])**2. / self.drive_I

        Px_per_drive_I = np.sum(normed_Ix,axis=-1) / self.sensor_size**2.
        Py_per_drive_I = np.sum(normed_Iy,axis=-1) / self.sensor_size**2.


        angles = np.arctan(Py_per_drive_I**0.5/Px_per_drive_I**0.5)
        return [angles, Px_per_drive_I, Py_per_drive_I]

    def powers_and_angels_no_interf(self,E1,E2):
        drive_I = np.abs(self.parameters['general']['drive_amp'])**2.

        normed_Ix = (np.abs(E1[0])**2. + np.abs(E2[0])**2.) / self.drive_I
        normed_Iy = (np.abs(E1[1])**2. + np.abs(E2[1])**2.) / self.drive_I

        Px_per_drive_I = np.sum(normed_Ix,axis=-1) / self.sensor_size**2.
        Py_per_drive_I = np.sum(normed_Iy,axis=-1) / self.sensor_size**2.


        angles = np.arctan(Py_per_drive_I**0.5/Px_per_drive_I**0.5)
        return [angles, Px_per_drive_I, Py_per_drive_I]


class FittingTools(object):

    def __init__(self,
        obs_points=None,
        param_file=None,
        **kwargs):
        """
        Args:
            obs_points: 3 element list (in legacy format of eye),
            in units of nm.

                obs_points[0]: list of points as rows
                obs_points[1]: meshed X array
                obs_points[2]: meshed Y array
        """

        if obs_points is None:
            ## Check for given self.exp_resolution
            if param_file is not None:
                ## Load resolution from parameter file
                self.parameters = load_param_file(param_file)
                # image grid resolution
                self.exp_resolution = self.parameters['optics']['sensor_pts']
                self.sensor_size = self.parameters['optics']['sensor_size']*cm_per_nm

                ## Define coordinate domain from center of edge pixels
                image_width_pixel_cc = ( ## Image width minus 1 pixel
                    self.sensor_size - self.sensor_size/self.exp_resolution)

                obs_points = diffi.observation_points(
                    x_min=-image_width_pixel_cc/2,
                    x_max=image_width_pixel_cc/2,
                    y_min=-image_width_pixel_cc/2,
                    y_max=image_width_pixel_cc/2,
                    points=self.exp_resolution
                    )

            elif param_file is None:
                raise ValueError(
                    "Must provide 'obs_points'"+
                    " or 'param_file' argument"
                    )
            else:
                raise ValueError('No obs_points or param?')

            self.obs_points = obs_points
        else:
            ## store given
            self.obs_points = obs_points


    def twoD_Gaussian(self,
        X, ## tuple of meshed (x,y) values
        amplitude,
        xo,
        yo,
        sigma_x,
        sigma_y,
        theta,
        offset,
        ):

        xo = float(xo)
        yo = float(yo)
        a = (
            (np.cos(theta)**2)/(2*sigma_x**2)
            +
            (np.sin(theta)**2)/(2*sigma_y**2)
            )
        b = (
            -(np.sin(2*theta))/(4*sigma_x**2)
            +
            (np.sin(2*theta))/(4*sigma_y**2)
            )
        c = (
            (np.sin(theta)**2)/(2*sigma_x**2)
            +
            (np.cos(theta)**2)/(2*sigma_y**2)
            )
        g = (
            offset
            +
            amplitude*np.exp( - (a*((X[0]-xo)**2) + 2*b*(X[0]-xo)*(X[1]-yo)
            +
            c*((X[1]-yo)**2)))
            )
        return g.ravel()

    def misloc_data_minus_model(
        self,
        fit_params,
        *normed_raveled_image_data,
        ):
        ''' fit gaussian to data '''
        gaus = self.twoD_Gaussian(
            (self.obs_points[1]/cm_per_nm, self.obs_points[2]/cm_per_nm),
            *fit_params ## ( A, xo, yo, sigma_x, sigma_y, theta, offset)
            )

        return gaus - normed_raveled_image_data

    def calculate_max_xy(self, images):
        ## calculate index of maximum in each image.
        apparent_centroids_idx = images.argmax(axis=-1)
        ## define locations for each maximum in physical coordinate system

        x_cen = (self.obs_points[1]/cm_per_nm).ravel()[apparent_centroids_idx]
        y_cen = (self.obs_points[2]/cm_per_nm).ravel()[apparent_centroids_idx]

        return [x_cen,y_cen]

    def calculate_apparent_centroids(self, images):
        """ calculate index of maximum in each image. """
        num_of_images = images.shape[0]

        apparent_centroids_xy = np.zeros((num_of_images,2))

        max_positions = self.calculate_max_xy(images)

        for i in np.arange(num_of_images):
            x0 = max_positions[0][i]
            y0 = max_positions[1][i]

            params0 = [1, x0, y0, 100, 100, 0, 0]

            args=tuple(images[i]/np.max(images[i]))
            fit_gaussian = opt.least_squares(
                self.misloc_data_minus_model, params0, args=args)
            resulting_fit_params = fit_gaussian['x']
            fit_result = self.twoD_Gaussian(
                (self.obs_points[1]/cm_per_nm, self.obs_points[2]/cm_per_nm), ## tuple of meshed (x,y) values
                *resulting_fit_params
                )
            centroid_xy = resulting_fit_params[1:3]
            apparent_centroids_xy[i] = centroid_xy
        ## define locations for each maximum in physical coordinate system

        return apparent_centroids_xy.T  ## returns [x_cen(s), y_cen(s)]

    def image_from_E(self, E):
        normed_I = np.sum(np.abs(E)**2., axis=0) / self.drive_I

        return normed_I

    def bin_ndarray(self, ndarray, new_shape, operation='mean'):
        """
        Used for averaging model image across experimental pixels.

        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.

        Number of output dimensions must match number of input dimensions and
            new axes must divide old ones.

        Example
        -------
        >>> m = np.arange(0,100,1).reshape((10,10))
        >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        >>> print(n)

        [[ 22  30  38  46  54]
         [102 110 118 126 134]
         [182 190 198 206 214]
         [262 270 278 286 294]
         [342 350 358 366 374]]

        source: https://stackoverflow.com/a/29042041

        """
        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                               new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                      ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
        return ndarray


    def rebin(self, a, shape):
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)


class PlottableDipoles(DipoleProperties):

    # Custom colormap to match Curly's, followed tutorial at
    # https://matplotlib.org/gallery/color/custom_cmap.html#sphx-glr-gallery-color-custom-cmap-py
    from matplotlib.colors import LinearSegmentedColormap

    curly_colors = [
        [164/255, 49/255, 45/255],
        [205/255, 52/255, 49/255],
        [223/255, 52/255, 51/255],
        [226/255, 52/255, 51/255],
        [226/255, 55/255, 51/255],
        [227/255, 60/255, 52/255],
        [228/255, 66/255, 52/255],
        [229/255, 74/255, 53/255],
        [230/255, 88/255, 53/255],
        [232/255, 102/255, 55/255],
        [235/255, 118/255, 56/255],
        [238/255, 132/255, 57/255],
        [242/255, 151/255, 58/255],
        [244/255, 165/255, 59/255],
        [248/255, 187/255, 59/255],
        [250/255, 223/255, 60/255],
        [195/255, 178/255, 107/255],
        [170/255, 160/255, 153/255],
        ]

    curlycm = LinearSegmentedColormap.from_list(
        'curly_cmap',
        curly_colors[::-1],
        N=500
        )

    a_shade_of_green = [0/255, 219/255, 1/255]

    def __init__(self,
        obs_points=None,
        **kwargs
        ):
        """ Establish dipole properties as atributes for reference in plotting
            functions.

            11/21/19: Taking out lineage to DipoleProperties, because it wasnt
            neccesary and confusing down the line
            """

        DipoleProperties.__init__(self,
            **kwargs)
        ## Load nanoparticle radii from parameter file
        # if param_file is not None:
        #     parameters = load_param_file(param_file)
        #     self.a_long_meters = parameters['plasmon']['fit_a1']*cm_per_nm
        #     self.a_short_meters = parameters['plasmon']['fit_a2']*cm_per_nm
        #     self.true_a_un_me = parameters['plasmon']['true_a_unique']*cm_per_nm
        #     self.true_a_de_me = parameters['plasmon']['true_a_degen']*cm_per_nm
        # elif a_long_meters is not None and a_short_meters is not None:
        #     self.a_long_meters = a_long_meters * cm_per_nm
        #     self.a_short_meters = a_short_meters * cm_per_nm
        # else: raise ValueError('Need param_file or both radii')

        ## Define geometry of NP in plot/focal plane
        self.el_c = self.a_short_meters / cm_per_nm
        ## Determine if disk or rod
        if self.a_long_meters > self.a_short_meters:
            ## Assume rod, so unique radius is in plane
            self.particle_shape = 'rod'
            self.el_a = self.a_long_meters / cm_per_nm

        elif self.a_long_meters <= self.a_short_meters:
            ## Assume disk, so unique radius is out of plot plane
            self.particle_shape = 'disk'
            self.el_a = self.a_short_meters / cm_per_nm

        ## Define image domain for plotting from parameter file
        ## Load resolution from parameter file
        if self.parameters is not None:
            # image grid resolution
            self.sensor_size = self.parameters['optics']['sensor_size']*cm_per_nm

            ## Define coordinate domain from center of edge pixels. Must be
            ## multiple of exp resolution for pixel averaging to work
            self.plot_resolution = 10 * self.exp_resolution
            image_width_pixel_cc = ( ## Image width minus 1 pixel
                self.sensor_size - self.sensor_size/self.plot_resolution)

            self.plt_obs_points = diffi.observation_points(
                x_min=-image_width_pixel_cc/2,
                x_max=image_width_pixel_cc/2,
                y_min=-image_width_pixel_cc/2,
                y_max=image_width_pixel_cc/2,
                points=self.plot_resolution
                )

        elif self.parameters is None:
            if obs_points is not None:
                self.plt_obs_points = obs_points
            else:
                raise ValueError(
                    "No 'param_file' or 'obs_pts' given to PlottableDipoles"+
                    "I havn't implemented a way to handle that..."
                    )
        else:
            raise ValueError('No obs_points or param?')
    def connectpoints(self, cen_x, cen_y, mol_x, mol_y, p, ax=None, zorder=1):
        x1, x2 = mol_x[p], cen_x[p]
        y1, y2 = mol_y[p], cen_y[p]
        if ax is None:
            plt.plot([x1,x2],[y1,y2],'k-', linewidth=.3, zorder=zorder)
        else:
            ax.plot([x1,x2],[y1,y2],'k-', linewidth=.3, zorder=zorder)

    def scatter_centroids_wLine(
        self,
        x_mol_loc,
        y_mol_loc,
        appar_cents,
        ax=None):

        x, y = appar_cents

        x_plot = x
        y_plot = y

        if ax is None:
            plt.figure(dpi=300)
            for i in np.arange(x_plot.shape[0]):
                self.connectpoints(
                    cen_x=x_plot,
                    cen_y=y_plot,
                    mol_x=x_mol_loc,
                    mol_y=y_mol_loc,
                    p=i,
                    zorder=3,
                    )

            localization_handle = plt.scatter(
                x_plot,
                y_plot,
                s=10,
                c=[PlottableDipoles.a_shade_of_green],
                zorder=4,
                )
            # plt.tight_layout()

        else:
            for i in np.arange(x_plot.shape[0]):
                self.connectpoints(
                    cen_x=x_plot,
                    cen_y=y_plot,
                    mol_x=x_mol_loc,
                    mol_y=y_mol_loc,
                    p=i,
                    ax=ax,
                    zorder=3,
                    )
            localization_handle = ax.scatter(
                x_plot,
                y_plot,
                s=10,
                c=[PlottableDipoles.a_shade_of_green],
                zorder=4,
                )
            return ax


    def quiver_plot(
        self,
        x_plot,
        y_plot,
        angles,
        plot_limits=[-25,550],
        title=r'Apparent pol. per mol. pos.',
        true_mol_angle=None,
        nanorod_angle=0,
        given_ax=None,
        plot_ellipse=True,
        cbar_ax=None,
        cbar_label_str=None,
        draw_quadrant=True,
        arrow_colors=None,
        ):
        """ Build quiver plot of fits """

        if given_ax is None:
            fig, (ax0, ax_cbar) = plt.subplots(
                nrows=1,ncols=2, figsize=(3.25,3), dpi=300,
                gridspec_kw = {'width_ratios':[6, 0.5]}
                )
        else:
            ax0 = given_ax

        # cmap = mpl.cm.nipy_spectral

        # If true angles are given as arguments, mark them
        if true_mol_angle is not None:

            ## Check to make sure angles are 1D array (assumes in-plane molecules)
            if (np.asarray(true_mol_angle).ndim <= 1):

                ## mark true orientation
                quiv_tr = ax0.quiver(
                    x_plot, y_plot, np.cos(true_mol_angle),np.sin(true_mol_angle),
                    color='black',
                    width=0.005,
                    scale=15,
                    scale_units='width',
                    pivot='mid',
                    headaxislength=0.0,
                    headlength=0.0,
                    zorder=1
                    )
            elif true_mol_angle.ndim == 2:
                ## Assume molecule out of plane, simply mark with point
                # quiv_tr = ax0.scatter(
                #     x_plot, y_plot,
                #     color='black',
                #     # scale=15,
                #     )
                pass
                ## Marking is done below

        cmap = PlottableDipoles.curlycm
        clim = [0, np.pi/2]

        if arrow_colors is None:
            ## Match arrow colors to angle in plane from horizontal

            # For main quiver, plot relative mispolarization if true angle is given
            if true_mol_angle is not None:
                diff_angles = np.abs(angles - true_mol_angle)
            else:
                diff_angles = np.abs(angles)

            arrow_colors = diff_angles
            # clim = [0, np.pi/2]


        elif type(arrow_colors) is np.ndarray:
            ## Assume arrow colors are being encoded by angle from optical axis
            # clim = [0, np.pi/2]
            # cmap = PlottableDipoles.curlycm.reversed()
            pass

        elif arrow_colors == 'gray':
            arrow_colors = [clim[0],] * len(angles)

        # print(f'arrow_colors = {arrow_colors}')
        ## Mark apparent orientation
        quiv_ap = ax0.quiver(
            x_plot,
            y_plot,
            np.cos(angles),
            np.sin(angles),
            arrow_colors,
            cmap=cmap,
            clim=clim,
            width=0.01,
            scale=12,
            scale_units='width',
            pivot='mid',
            zorder=2,
            headaxislength=2.5,
            headlength=2.5,
            headwidth=2.5,
            )

        ## Mark molecule locations
        scat_tr = ax0.scatter(x_plot, y_plot, s=3,
            color='black',
            zorder=5,
            )

        ax0.axis('equal')
        ax0.set_xlim(plot_limits)
        ax0.set_ylim(plot_limits)
        ax0.set_title(title)
        ax0.set_xlabel(r'$x$ [nm]')
        ax0.set_ylabel(r'$y$ [nm]')

        # Build colorbar if building single Axes figure
        if given_ax is None:
            if cbar_ax is None:
                cbar_ax=ax_cbar
            if cbar_label_str is None:
                cbar_label_str=r'observed angle $\phi$'

            cb = self.build_colorbar(
                cbar_ax=cbar_ax,
                cbar_label_str=cbar_label_str,
                cmap=cmap
                )
        else: # Don't build colorbar
            pass

        if nanorod_angle == np.pi/2:

            if draw_quadrant==True:

                # rod = self.draw_rod(color=curly_nanorod_color_light)

                particle_patches = self.real_particle_quad_patches()

                for piece in particle_patches:
                    ax0.add_patch(piece)
                # ax0.add_patch(top_wedge)
                # ax0.add_patch(rect)


            elif draw_quadrant==False:
                # Draw rod
                [circle, rect, bot_circle] = self.draw_rod()

                ax0.add_patch(circle)
                ax0.add_patch(rect)
                ax0.add_patch(bot_circle)

        ## Draw projection of model spheroid as ellipse
        if plot_ellipse==True:
            pass
            # curly_dashed_line_color = (120/255, 121/255, 118/255)

            ## Uncomment for drawing model spheroid quadrant
            # if draw_quadrant is True:
            #     ellip_quad = mpl.patches.Arc(
            #         (0,0),
            #         2*self.el_c,
            #         2*self.el_a,
            #         # angle=nanorod_angle*180/np.pi,
            #         theta1=0,
            #         theta2=90,
            #         # fill=False,
            #         # edgecolor='Black',
            #         edgecolor=curly_dashed_line_color,
            #         linestyle='--',
            #         linewidth=1.5
            #         )
            #     ax0.add_patch(ellip_quad)

                # translucent_ellip = mpl.patches.Ellipse(
                #     (0,0),
                #     2*self.el_a,
                #     2*self.el_c,
                #     angle=nanorod_angle*180/np.pi,
                #     fill=False,
                #     # edgecolor='Black',
                #     edgecolor=curly_dashed_line_color,
                #     linestyle='--',
                #     alpha=0.5
                #     )
                # ax0.add_patch(translucent_ellip)


                # # Draw lines along x and y axis to finish bounding
                # # quadrant.
                # ax0.plot(
                #     [0,0],
                #     [0,self.el_a],
                #     linestyle='--',
                #     color=curly_dashed_line_color,
                #     )
                # ax0.plot(
                #     [0,self.el_c],
                #     [0,0],
                #     linestyle='--',
                #     color=curly_dashed_line_color,
                #     )


            # elif draw_quadrant is False:
            #     ellip = mpl.patches.Ellipse(
            #         (0,0),
            #         2*self.el_a,
            #         2*self.el_c,
            #         angle=nanorod_angle*180/np.pi,
            #         fill=False,
            #         # edgecolor='Black',
            #         edgecolor=curly_dashed_line_color,
            #         linestyle='--',
            #         )

            #     ax0.add_patch(ellip)

        elif plot_ellipse==False:
            pass

        quiver_axis_handle = ax0
        return [quiver_axis_handle]

    def real_particle_quad_patches(
        self,
        **kwargs):

        if self.a_long_meters > self.a_short_meters:
            top_wedge = mpl.patches.Wedge(
                center=(0, 24),
                r=20,
                theta1=0,
                theta2=90,
                facecolor=curly_nanorod_color,
                edgecolor='Black',
                linewidth=0,
                )
            rect = mpl.patches.Rectangle(
                (0, 0),
                20,
                24,
                angle=0.0,
                # facecolor='Gold',
                facecolor=curly_nanorod_color,
                edgecolor='Black',
                linewidth=0,
                )

            return [top_wedge, rect]

        elif self.a_long_meters <= self.a_short_meters:
            top_wedge = mpl.patches.Wedge(
                center=(0, 0),
                r=self.true_a_de_me/cm_per_nm,
                theta1=0,
                theta2=90,
                facecolor=curly_nanorod_color,
                edgecolor='Black',
                linewidth=0,
                )
            # rect = mpl.patches.Rectangle(
            #     (0, 0),
            #     20,
            #     24,
            #     angle=0.0,
            #     # facecolor='Gold',
            #     facecolor=curly_nanorod_color,
            #     edgecolor='Black',
            #     linewidth=0,
            #     )

            return [top_wedge,]

    def draw_rod(
        self,
        color=None,
        **kwargs
        ):

        if color is None:
            color = self.curly_nanorod_color
        circle = mpl.patches.Circle(
            (0, 24),
            20,
            # facecolor='Gold',
            facecolor=color,
            edgecolor='Black',
            linewidth=0,
            )
        bot_circle = mpl.patches.Circle(
            (0, -24),
            20,
            # facecolor='Gold',
            facecolor=color,
            edgecolor='Black',
            linewidth=0,
            )
        rect = mpl.patches.Rectangle(
            (-20,-24),
            40,
            48,
            angle=0.0,
            # facecolor='Gold',
            facecolor=color,
            edgecolor='Black',
            linewidth=0,
            )

        return [circle, rect, bot_circle]

    def build_colorbar(self, cbar_ax, cbar_label_str, cmap):

        color_norm = mpl.colors.Normalize(vmin=0, vmax=np.pi/2)

        cb1 = mpl.colorbar.ColorbarBase(
            ax=cbar_ax,
            cmap=cmap,
            norm=color_norm,
            orientation='vertical',
            )
        cb1.set_label(cbar_label_str)

        cb1.set_ticks([0, np.pi/8, np.pi/4, np.pi/8 * 3, np.pi/2])
        cb1.set_ticklabels(
            [r'$0$', r'$22.5$',r'$45$',r'$67.5$',r'$90$']
            )

        return cb1


    def calculate_mislocalization_magnitude(self, x_cen, y_cen, x_mol, y_mol):
        misloc = ( (x_cen-x_mol)**2. + (y_cen-y_mol)**2. )**(0.5)
        return misloc


class CoupledDipoles(PlottableDipoles, FittingTools):

    def __init__(self, **kwargs):
        """ Container for methods which perform coupled dipole dynamics
            calculations contained in 'coupled_dipoles' module as well
            as field calculations.

            Inherits initialization from;
                DipoleProperties: define dipole properties from
                    parameter file or given as arguments and
                    store as instance attributes.
                PlottableDipoles: Not sure what this is used for here, I
                    know it defines the quenching zone as class
                    attrubtes.
                FittingTools: Initialized 'obs_points' as instance
                    attribute.
            """
        # DipoleProperties.__init__(self, **kwargs)
        ## Send obs_points or param_file to FittingTools
        FittingTools.__init__(self, **kwargs)
        ## Regardless, will establish self.obs_points

        ## Get plotting methods
        PlottableDipoles.__init__(self, **kwargs)

    def foc_dif_dip_fields(self, dipole_mag_array, dipole_coordinate_array):
        ''' Evaluates analytic form of focused+diffracted dipole fields
            anlong observation grid given

            Args
            ----
                dipole_mag_array: array of dipole moment vecotrs
                    with shape ~(n dipoles, 3 cartesean components)
                dipole_coordinate_array: same shape structure but
                    the locations of the dipoles.

            Returns
            -------
                Fields with shape ~ (3, ?...)
            '''

        p = dipole_mag_array
        bfx = dipole_coordinate_array

        v_rel_obs_x_pts = (self.obs_points[1].ravel()[:,None] - bfx.T[0]).T
        v_rel_obs_y_pts = (self.obs_points[2].ravel()[:,None] - bfx.T[1]).T

        px_fields = afi.E_field(
            0,
            v_rel_obs_x_pts,
            v_rel_obs_y_pts,
            (self.drive_energy_eV/hbar)*np.sqrt(self.eps_b)/c
            )
        py_fields = afi.E_field(
            np.pi/2,
            v_rel_obs_x_pts,
            v_rel_obs_y_pts,
            (self.drive_energy_eV/hbar)*np.sqrt(self.eps_b)/c
            )
        pz_fields = afi.E_pz(
            xi=v_rel_obs_x_pts,
            y=v_rel_obs_y_pts,
            k=(self.drive_energy_eV/hbar)*np.sqrt(self.eps_b)/c
            )

        ## returns [Ex, Ey, Ez] for dipoles oriented along cart units

        Ex = (
            p[:,0,None]*px_fields[0]
            +
            p[:,1,None]*py_fields[0]
            +
            p[:,2,None]*pz_fields[0]
            )
        Ey = (
            p[:,0,None]*px_fields[1]
            +
            p[:,1,None]*py_fields[1]
            +
            p[:,2,None]*pz_fields[1]
            )
        Ez = (
            p[:,0,None]*px_fields[2]
            +
            p[:,1,None]*py_fields[2]
            +
            p[:,2,None]*pz_fields[2]
            )

        return np.array([Ex,Ey,Ez])


    def dipole_fields(
        self,
        locations,
        mol_angle=0,
        plas_centroid=np.array([[0, 0, 0]]),
        plas_angle=np.pi/2,
        ):
        """ Calculate image fields of coupled plasmon and molecule
            dipole.

            Args
            ----


            Returns
            -------
            """
        ## rel distance to mol
        d = locations * cm_per_nm
        ## plasmon locatiuon
        plas_loc = plas_centroid * cm_per_nm
        ## Get dipole mognitudes
        p0, p1 = cp.dipole_mags_gened(
            mol_angle,
            plas_angle,
            d_col=d,
            E_d_angle=None,
            drive_hbar_w=self.drive_energy_eV,
            alpha0_diag=self.alpha0_diag_dyad,
            alpha1_diag=self.alpha1_diag_dyad,
            n_b=np.sqrt(self.eps_b),
            drive_amp=self.drive_amp,
            )
        mol_E = self.foc_dif_dip_fields(
            dipole_mag_array=p0,
            dipole_coordinate_array=d+plas_loc,
            )
        plas_E = self.foc_dif_dip_fields(
            dipole_mag_array=p1,
            dipole_coordinate_array=plas_loc,
            )

        p0_unc, = cp.uncoupled_p0(
            mol_angle,
            E_d_angle=None,
            alpha_0_p0=self.alpha0_diag_dyad,
            drive_amp=self.drive_amp,
            )

        # if type(mol_angle)==np.ndarray and mol_angle.shape[0]>1:
        p0_unc_E = self.foc_dif_dip_fields(
            dipole_mag_array=np.atleast_2d(p0_unc),
            dipole_coordinate_array=d
            )
        # elif (
        #     type(mol_angle) == int or
        #     type(mol_angle) == float or
        #     type(mol_angle) == np.float64 or
        #     (type(mol_angle) == np.ndarray and mol_angle.shape[0]==1)
        #     ):
        #     p0_unc_E = self.foc_dif_dip_fields(
        #         dipole_mag_array=p0_unc[None,:],
        #         dipole_coordinate_array=d,
        #         )
        return [mol_E, plas_E, p0_unc_E, p0, p1]


class MolCoupNanoRodExp(CoupledDipoles, BeamSplitter):
    ''' Class responsible for modeling and plasmon-enhances single-
        molecule imaging experiment.
        '''
    ## set up inverse mapping from observed -> true angle for signle molecule
    ## in the plane.
    saved_mapping = np.loadtxt(txt_file_path+'/obs_pol_vs_true_angle.txt')
    true_ord_angles, obs_ord_angles = saved_mapping.T
    #from scipy import interpolate
    f = interpolate.interp1d(true_ord_angles, obs_ord_angles)
    f_inv = interpolate.interp1d(
        obs_ord_angles[:251],
        true_ord_angles[:251],
        bounds_error=False,
        fill_value=(0,np.pi/2)
        )

    def __init__(
        self,
        locations,
        mol_angle=0,
        plas_centroid=np.array([[0, 0, 0]]),
        plas_angle=np.pi/2,
        for_fit=False,
        exclude_interference=False,
        auto_calc_fiels=True,
        **kwargs
        ):
        """
            Args:
                locations: list of cartesien coordinates of molecules
                    in cm of shape = (number of molecules, 3)
                mol_angle: if float of 1D array, interpreted as the
                    angle of molecule in focal plane relative to x-axis
                    . If 2D array of size (number of locations, 2),
                    interpreted as a list of (theta, phi) coordinate pairs for each
                plas_angle:
                    angle of dominent dipole plasmon, for a prolate spheroid this is
                    the long axis.
                for_fit:  simply turns off auto quenching within defined
                    quenching region.
            """
        # Set up instance attributes
        self.exclude_interference = exclude_interference
        self.mol_locations = locations
        self.mol_angles = mol_angle
        self.plas_centroid = plas_centroid
        self.rod_angle = plas_angle

        ## Send param_file or specified dipole params to
        CoupledDipoles.__init__(self,
            **kwargs
            )
        ## define incident intensity as inst attr.
        BeamSplitter.__init__(self, **kwargs)

        # Filter out molecules in region of fluorescence quenching
        # self.el_a and self.el_c are now defined inside PlottingTools
        # __init__().
        # self.el_a = self.a_long_meters / cm_per_nm
        # self.el_c = self.a_short_meters / cm_per_nm
        #
        # define quenching region
        self.quench_radius_a_nm = self.el_a + self.fluo_quench_range
        self.quench_radius_c_nm = self.el_c + self.fluo_quench_range
        self.input_x_mol = locations[:,0]
        self.input_y_mol = locations[:,1]
        self.pt_is_in_ellip = self.mol_not_quenched()
        ## select molecules outside region,
        if for_fit==False:
            self.mol_locations = locations[self.pt_is_in_ellip]
            ## select molecule angles if listed per molecule,
            if type(mol_angle)==np.ndarray and mol_angle.shape[0]>1:
                self.mol_angles = mol_angle[self.pt_is_in_ellip]
            else: self.mol_angles = mol_angle
        elif for_fit==True:
            self.mol_locations = locations
            self.mol_angles = mol_angle

        if auto_calc_fiels:
            self.calculate_fields()

        # Calculate plot domain from molecule locations
        self.default_plot_limits = [
            (
                np.min(self.mol_locations)
                -
                (
                    (
                        np.max(self.mol_locations)
                        -
                        np.min(self.mol_locations)
                        )*.1
                    )
                ),
            (
                np.max(self.mol_locations)
                +
                (
                    (
                        np.max(self.mol_locations)
                        -
                        np.min(self.mol_locations)
                        )*.1
                    )
                ),
            ]

    def calculate_fields(self):
        # Automatically calculate fields with coupled dipoles upon
        # instance initialization.
        (
            self.mol_E,
            self.plas_E,
            self.p0_unc_E,
            self.p0,
            self.p1
            ) = self.dipole_fields(
                locations=self.mol_locations,
                mol_angle=self.mol_angles,
                plas_centroid=self.plas_centroid,
                plas_angle=self.rod_angle,
                )

        # Calcualte images
        self.anal_images = self.image_from_E(self.mol_E + self.plas_E)

    def work_on_rod_by_mol(self,
        locations=None,
        mol_angle=None,
        plas_angle=None,
        ):
        """ Calculate interaction energy by evaluating
            - p_rod * E_mol = - p_rod * G * p_mol
            """
        # Set instance attributes as default
        if locations is None:
            locations = self.mol_locations
        if mol_angle is None:
            mol_angle = self.mol_angles
        if plas_angle is None:
            plas_angle = self.rod_angle

        d = locations*cm_per_nm

        Gd = cp.G(self.drive_energy_eV, d, np.sqrt(self.eps_b))

        Gd_dot_p0 = np.einsum('...ij,...j->...i', Gd, self.p0)

        p1stardot_dot_E0 = np.einsum(
            'ij,ij->i',
            -1j*self.drive_energy_eV/hbar * self.p1,
            Gd_dot_p0
            )

        work_done = 1/2 * np.real(p1stardot_dot_E0)
        return work_done


    def calculate_localization(self, save_fields=True):
        """ """
        FitModelToData.calculate_localization(self, save_fields=save_fields)


    def calculate_polarization(self):
        """ Calculate polarization with beam splitter """

        # Calculate fields and angles and assign as instance attribute
        if hasattr(self, 'mol_E') and hasattr(self, 'plas_E'):
            if self.exclude_interference == False:
                self.angles, self.Px_per_drive_I, self.Py_per_drive_I = (
                    self.powers_and_angels(
                        self.mol_E + self.plas_E
                        )
                    )
            # For exclusion of interference, fields must be input
            # seperately into funtion 'powers_and_angels_no_interf'.
            elif self.exclude_interference == True:
                self.angles, self.Px_per_drive_I, self.Py_per_drive_I = (
                    self.powers_and_angels_no_interf(
                        self.mol_E,
                        self.plas_E
                        )
                    )

        # Simulation instance will have field namd differently and
        # must be transposed.
        elif hasattr(self, 'bem_E'):
            # Can not extract interference from simulation
            self.angles, self.Px_per_drive_I, self.Py_per_drive_I = (
                    self.powers_and_angels(
                        np.transpose(self.bem_E, (2,0,1))
                        )
                )
        self.mispol_angle = MolCoupNanoRodExp.f_inv(self.angles)


    def mol_not_quenched(self,
        rod_angle=None,
        input_x_mol=None,
        input_y_mol=None,
        long_quench_radius=None,
        short_quench_radius=None,
        ):
        '''Given molecule location as ('input_x_mol', 'input_y_mol'),
            returns molecule locations that are outside the fluorescence
            quenching zone, defined as 10 nm from surface of fit spheroid
            '''

        if rod_angle is None:
            rod_angle=self.rod_angle

        if input_x_mol is None:
            input_x_mol=self.input_x_mol

        if input_y_mol is None:
            input_y_mol=self.input_y_mol

        if long_quench_radius is None:
            long_quench_radius=self.quench_radius_a_nm

        if short_quench_radius is None:
            short_quench_radius=self.quench_radius_c_nm

        rotated_x = (
            np.cos(rod_angle)*input_x_mol
            + np.sin(rod_angle)*input_y_mol
            )
        rotated_y = (
            -np.sin(rod_angle)*input_x_mol
            + np.cos(rod_angle)*input_y_mol
            )

        rotated_ellip_eq = (
            rotated_x**2./long_quench_radius**2
            + rotated_y**2./short_quench_radius**2
            )

        return (rotated_ellip_eq > 1)


    def plot_mispol_map(self,
        plot_limits=None,
        given_ax=None,
        plot_ellipse=True,
        draw_quadrant=True,
        plot_mispol_map=None,
        arrow_colors=None,
        ):

        if plot_limits is None: plot_limits = self.default_plot_limits
        if not hasattr(self, 'mispol_angle'):
            self.calculate_polarization()

        if np.asarray(np.atleast_1d(self.mol_angles).ndim) < 2:
            mol_angles = self.mol_angles
        else:
            mol_angles = None

        quiv_ax, = self.quiver_plot(
            x_plot=self.mol_locations[:,0],
            y_plot=self.mol_locations[:,1],
            angles=self.mispol_angle,
            plot_limits=plot_limits,
            arrow_colors=arrow_colors,
            true_mol_angle=mol_angles,
            nanorod_angle=self.rod_angle,
            title=r'Split Pol. and Gau. Fit Loc.',
            given_ax=given_ax,
            plot_ellipse=plot_ellipse,
            draw_quadrant=draw_quadrant,
            )

        return quiv_ax

    def plot_mispol_map_wMisloc(self,
        plot_limits=None,
        given_ax=None,
        plot_ellipse=True,
        draw_quadrant=True,
        arrow_colors=None,
        ):

        # Compulate localizations if not already stored as cclass attrubute
        if not hasattr(self, 'appar_cents'):
            self.calculate_localization()

        # Set default plot limits if not specified
        if plot_limits is None: plot_limits = self.default_plot_limits

        # Plot mispolarizations
        quiv_ax = self.plot_mispol_map(
            plot_limits,
            given_ax=given_ax,
            plot_ellipse=plot_ellipse,
            draw_quadrant=draw_quadrant,
            arrow_colors=arrow_colors,
            )

        # Plot mislocalizations
        self.scatter_centroids_wLine(
            self.mol_locations[:,0],
            self.mol_locations[:,1],
            self.appar_cents,
            quiv_ax,
            )
        return quiv_ax

    def plot_mislocalization_magnitude_correlation(self):
        if not hasattr(self, 'appar_cents'):
            self.calculate_localization()
        self.misloc_mag = calculate_mislocalization_magnitude(
            self.x_cen,
            self.y_cen,
            self.mol_locations[:,0],
            self.mol_locations[:,1],
            )

        plt.figure(dpi=300)
        plt.scatter(self.misloc_mag, mispol_angle, s=10, c='Black', zorder=3)
        plt.tight_layout()
        plt.xlabel('Magnitude of mislocalization [nm]')
        plt.ylabel('Apparent angle [deg]')
        plt.yticks([0,  np.pi/8,  np.pi/4, np.pi/8 *3, np.pi/2],
                   ['0','22.5','45','57.5','90'])
        return plt.gca()

    def plot_fields(self, ith_molecule):
        plt.figure(figsize=(3,3),dpi=600)
        plt.pcolor(
            self.obs_points[1]/cm_per_nm,
            self.obs_points[2]/cm_per_nm,
            (
                self.anal_images[ith_molecule,:]
                ).reshape(self.obs_points[1].shape)
            )
        plt.colorbar()
        plt.title(r'$|E|^2/|E_\mathrm{inc}|^2$')
        plt.xlabel(r'$x$ [nm]')
        plt.ylabel(r'$y$ [nm]')
        # plt.quiver(self.mol_locations[ith_molecule, 0], self.mol_locations[ith_molecule, 1],
                   # np.cos(self.mol_angles[ith_molecule]),np.sin(self.mol_angles[ith_molecule]),
                   # color='white',pivot='middle')
        return plt.gca()


    def save_exp_inst_for_fit(self):
        """ Save's data needed for fitting to txt files for faster
            debugging and reproducability.
            """

        # Save Images
        if hasattr(self, 'BEM_images'):
            np.savetxt()
        # Save plot limits

        # save mol locations

        # save mol angles

        # save nanorod angle


class FitModelToData(CoupledDipoles, BeamSplitter):
    ''' Class to contain fitting functions that act on class 'MolCoupNanoRodExp'
    as well as variables that are needed by 'MolCoupNanoRodExp'

    Takes any kwargs that get sent to DipoleProperties.
    '''
    def __init__(self,
        image_data,
        plas_centroids=None,
        rod_angle=None,
        **kwargs
        ):
        ## Store all input args for later
        self.mol_angles=0

        ## Rod angle is a given, for a disk this doesnt matter
        if rod_angle is None:
            self.rod_angle = np.pi/2
        else:
            self.rod_angle = rod_angle

        ## Images and presumed plasmon centroid
        self.image_data = image_data
        if plas_centroids is None:
            self.plas_centroids = np.zeros((image_data.shape[0], 2))
        else:
            self.plas_centroids = plas_centroids

        # # This should really be moved to the fit method...
        # self.ini_guess = ini_guess

        CoupledDipoles.__init__(self, **kwargs)
        BeamSplitter.__init__(self, **kwargs)

        ## Define quenching readii for smart initial guess,
        ## attributes inherited from DipoleProperties.
        if hasattr(self, 'quench_over_real_disk'):
            if self.quench_over_real_disk:
                self.quench_radius_a_nm = self.true_a_de_me / cm_per_nm
                self.quench_radius_c_nm = self.quench_radius_a_nm

        else:
            self.quench_radius_a_nm = self.el_a + self.fluo_quench_range
            self.quench_radius_c_nm = self.el_c + self.fluo_quench_range


    def fit_model_to_image_data(self,
        images=None,
        ini_guess=None,
        check_fit_loc=False,
        check_ini=False,
        max_fail_converge=10,
        let_mol_ori_out_of_plane=False,
        return_full_fit_output=False,
        integral_normalize=False,
        avg_model_over_pixels=False,
        least_squares_kwargs={},
        ):
        """ Returnes array of model fit parameters, unless
                'return_full_fit_output' == True
            then the result is that array followed by a list
            of the dictionaries returned by 'opt.least_squares'.
            """

        ## calculate index of maximum in each image,
        ## going to use this for the initial position guess
        if images is None:
            images = self.image_data
        ## initialize array to hold fit results for arbitrary number of images
        num_of_images = images.shape[0]
        if not let_mol_ori_out_of_plane:
            self.model_fit_results = np.zeros((num_of_images, 3))
        elif let_mol_ori_out_of_plane:
            self.model_fit_results = np.zeros((num_of_images, 4))
        ## If arg 'return_full_fit_output' is True, then initialize a list
        ## to hold outputs of fit routine. For 'opt.least_squares' this is
        ## a dict.
        if return_full_fit_output:
            self.full_model_fit_results = [0,] * num_of_images

        ## If going to use positions of max intensity as initial guess for
        ## molecule position, calculate positions
        if type(ini_guess) is np.ndarray:
            pass

        # If using Gaussian centroids, calculate.
        elif (ini_guess == 'Gauss') or (ini_guess == 'gauss') or (ini_guess == 'on_edge'):
            self.x_gau_cen_abs, self.y_gau_cen_abs = self.calculate_apparent_centroids(
                images
                )
        elif ini_guess is None:
            max_positions = self.calculate_max_xy(images)

        ## Loop through images and fit.
        for i in np.arange(num_of_images):
            print(f"\n")
            print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Fitting model to molecule {i}")
            print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            ## Establish initial guesses for molecules

            # If no initial guesses specified as kwarg, use pixel
            # location of maximum intensity.
            ## Assume ini_guesses given as numy array.
            if type(ini_guess) is np.ndarray:
                ini_x = ini_guess[i, 0]
                ini_y = ini_guess[i, 1]

            # If kwarg 'Gauss' specified, use centroid of gaussian
            # localization as inilial guess
            elif (ini_guess == 'Gauss') or (ini_guess == 'gauss'):
                ## Define relative to plasmon location
                ini_x, ini_y = [
                    self.x_gau_cen_abs[i] - self.plas_centroids[i, 0],
                    self.y_gau_cen_abs[i] - self.plas_centroids[i, 1]
                    ]

            elif ini_guess == 'on_edge':
                ini_x, ini_y = self._better_init_loc(
                    self.x_gau_cen_abs[i] - self.plas_centroids[i, 0],
                    self.y_gau_cen_abs[i] - self.plas_centroids[i, 1]
                    )

            elif ini_guess is None:
                ini_x = np.round(max_positions[0][i])
                ini_y = np.round(max_positions[1][i])

            print(
                'initial guess position: ({},{})'.format(
                    ini_x, ini_y
                    )
                )

            ## Randomize initial molecule oriantation, maybe do something
            ## smarter later.
            if not let_mol_ori_out_of_plane:
                ini_mol_orientation = np.pi * np.random.random(1)
                params0 = [ini_x, ini_y, ini_mol_orientation]
            elif let_mol_ori_out_of_plane:
                ini_mol_orientation = (
                    np.array([[np.pi/2, np.pi]]) * np.random.random((1, 2)))
                # And assign parameters for fit.
                params0 = [ini_x, ini_y, *ini_mol_orientation.ravel()]
            print(f"initial guess angle = {params0[2:]}")

            # Should test inital guess here, since I am only changing the
            # inital guess. Later loop on fitting could still be healpful later.
            if check_ini == True:
                print('Checking inital guess')
                ini_guess_not_quench = MolCoupNanoRodExp.mol_not_quenched(
                    self,
                    self.rod_angle,
                    ini_x,
                    ini_y,
                    self.quench_radius_a_nm,
                    self.quench_radius_c_nm,
                    )
                print(
                    # 'self.rod_angle, ', self.rod_angle, '\n',
                    # 'ini_x, ', ini_x, '\n',
                    # 'ini_y, ', ini_y, '\n',
                    '    self.quench_radius_a_nm, ', self.quench_radius_a_nm,
                    '    self.quench_radius_c_nm, ', self.quench_radius_c_nm,
                    )
                print('    In quenching zone? {}'.format(not ini_guess_not_quench))
                if ini_guess_not_quench:
                    # continure to fit
                    pass

                elif not ini_guess_not_quench:
                    # Adjust ini_guess to be outsie quenching zone
                    print(f'Initial guess in quench. Zone, OG params: {params0}')
                    on_edge_x, on_edge_y = self._better_init_loc(ini_x, ini_y)
                    ini_x += on_edge_x
                    ini_y += on_edge_y
                    params0[:2] = ini_x, ini_y
                    print(f'Params shifted to: {params0}')

            ## Normalize images for fitting.
            if integral_normalize:
                a_raveled_normed_image = images[i]/(
                    images[i].sum()
                    /
                    (self.sensor_size/cm_per_nm)**2. ## A in nm
                    )

            elif not integral_normalize:
                a_raveled_normed_image = images[i]/np.max(images[i])


            ## Place image data and plasmon location in tp tuple as required
            ## by `opt.least_squares`.
            fit_args = (
                a_raveled_normed_image,
                self.plas_centroids[i],
                integral_normalize
                )

            fit_kwargs = {}

            ## If averaging model across pixels add to fit_args
            if avg_model_over_pixels:
                fit_kwargs['avg_model_over_pixels'] = avg_model_over_pixels
            else:
                pass

            ## Run fit unitil satisfied with molecule position
            mol_pos_accepted = False
            fit_quenched_counter = 1
            fail_to_converge_counter = 1

            while mol_pos_accepted == False:
                print(f"running fit...")
                # print(f"self = {self}")
                ## Perform fit
                optimized_fit = opt.least_squares(
                    self._misloc_data_minus_model, ## residual
                    params0, ## initial guesses
                    args=fit_args, ## data to fit
                    kwargs=fit_kwargs,
                    **least_squares_kwargs
                    )

                ## Check for fit convergence and retry if it failed for a finite
                ## number of tries.
                if optimized_fit['success']:
                    print(f"SUCCESS, Resulting fit params: {optimized_fit['x']}")
                else:
                    ## try a few more times with new initial guesses
                    if fail_to_converge_counter < max_fail_converge:
                        print(
                            f"FAILURE, fit not converged, randomize angle "
                            +
                            f"guess and try again. Unconverged counter = "
                            +
                            f"{fail_to_converge_counter}."
                            )
                        fail_to_converge_counter += 1
                        ## Randomize angle
                        if not let_mol_ori_out_of_plane:
                            params0[2] = np.pi * np.random.random(1)
                        elif let_mol_ori_out_of_plane:
                            params0[2:] = (
                                np.array([np.pi/2, np.pi])
                                *
                                np.random.random(2)
                                )
                        continue
                    else:
                        print(
                            f"FAILURE to converge {fail_to_converge_counter} "
                            +
                            "times in a row, giving up."
                            )
                        break

                ## Break loop here if we don't want to iterate through smarter
                ## initial guesses.
                if check_fit_loc == False:
                    # PROCEED NO FURTHER
                    break
                elif check_fit_loc == True:
                    # Proceed to more fits
                    pass

                ## Check molecule postion from fit
                fit_loc = optimized_fit['x'][:2]
                ## True or false?
                fit_loc_quenched = not MolCoupNanoRodExp.mol_not_quenched(
                    self,
                    self.rod_angle,
                    fit_loc[0],
                    fit_loc[1],
                    self.quench_radius_a_nm,
                    self.quench_radius_c_nm,
                    )

                if fit_loc_quenched:
                    # Try fit again, but with a different initial guess.

                    # ~~~~~~~~~~~~~
                    # Add radius to initial guess.
                    on_edge_x, on_edge_y = self._better_init_loc(ini_x, ini_y)
                    ini_x += on_edge_x
                    ini_y += on_edge_y
                    params0[:2] = ini_x, ini_y

                    ## Randomize angle
                    if not let_mol_ori_out_of_plane:
                        params0[2] = np.pi * np.random.random(1)
                    elif let_mol_ori_out_of_plane:
                        params0[2:] = np.array([np.pi/2, np.pi]) * np.random.random(2)

                    print('fit quenched, ini guess now: {}'.format(params0))

                    print(f"Quench counter = {fit_quenched_counter}.")

                    fit_quenched_counter += 1

                    if fit_quenched_counter > 100:
                        ## Give up
                        mol_pos_accepted = True
                        print(
                            f"Giving up, fit pos. quenched but accepted")

                elif not fit_loc_quenched:
                    # Fit location is far enough away from rod to be
                    # reasonable
                    mol_pos_accepted = True
                    print(
                        f"Fit pos. ACCEPTED as unquenched: took "
                        +
                        f"{fit_quenched_counter}"
                        +
                        " fit(s).")

            # We satisfied apparently.
            # Store fit result parameters as class instance attribute.
            if optimized_fit['success']:
                self.model_fit_results[i][:2] = optimized_fit['x'][:2]
                # Project fit result angles to first quadrant
                if not let_mol_ori_out_of_plane:
                    angle_in_first_quad = self.map_angles_to_first_quad(
                        optimized_fit['x'][2]
                        )
                elif let_mol_ori_out_of_plane:
                    angle_in_first_quad = optimized_fit['x'][2:]

                self.model_fit_results[i][2:] = angle_in_first_quad

            elif not optimized_fit['success']:

                self.model_fit_results[i][:] = np.nan

            if return_full_fit_output:
                self.full_model_fit_results[i] = optimized_fit

        if not return_full_fit_output:
            return self.model_fit_results

        elif return_full_fit_output:
            return self.model_fit_results, self.full_model_fit_results


    def map_angles_to_first_quad(self, angles):
        angle_in_first_quad = np.arctan(
            np.abs(np.sin(angles))
            /
            np.abs(np.cos(angles))
            )
        return angle_in_first_quad

    # def map_angles_to_first_two_quads(self, angles):
    #     angle_in_first_quad = np.arctan(
    #         np.abs(np.sin(angles))
    #         /
    #         np.abs(np.cos(angles))
    #         )
    #     return angle_in_first_quad


    def calculate_localization(self, save_fields=True, images=None,):
        """ """
        if images is None:
            ## Make a selection from attributes already defined.
            if hasattr(self, 'BEM_images'):
                print("Calculating Gaussian centroid with BEM_images")
                images = self.BEM_images

            elif hasattr(self, 'anal_images'):
                print("Calculating Gaussian centroid with analytic images")
                images = self.anal_images

        self.appar_cents = self.calculate_apparent_centroids(
            images
            )
        # redundant, but I'm lazy and dont want to clean dependencies.
        self.x_gau_cen_abs, self.y_gau_cen_abs = self.appar_cents

        if save_fields == False:
            del self.mol_E
            del self.plas_E


    def _better_init_loc(self, ini_x, ini_y):
        """ Smarter initial guess algorithm if position of maxumum
            intensity fails to return molecule position outside the
            particle quenching zone. Not that fillting routine
            currently (03/22/19) loops through this algorith.
            """
        ## Move initial guess outside quenching zone.
        #
        # Convert position to polar coords
        circ_angl = afi.phi(ini_x, ini_y)
        # sub radius with ellipse radius at given angle
        radius = self._polar_ellipse_semi_r(circ_angl)
        # convert back to cartisean and shift guess outward by a radius
        smarter_ini_x, smarter_ini_y = self.circ_to_cart(radius, circ_angl)

        return smarter_ini_x, smarter_ini_y


    def _polar_ellipse_semi_r(self, phi):
        a = self.quench_radius_a_nm
        c = self.quench_radius_c_nm

        radius = a*c/np.sqrt(
            c**2. * np.sin(phi)**2.
            +
            a**2. * np.cos(phi)**2.
            )

        return radius


    def circ_to_cart(self, r, phi):
        x = r*np.cos(phi)
        y = r*np.sin(phi)

        return x, y


    def _misloc_data_minus_model(self,
        fit_params,
        *fit_args,
        **fit_kwargs,
        ):
        ''' fit image model to data.
            arguments;
                fit_params = [ini_x, ini_y, ini_mol_orintation]
                'ini_x' : units of nm from plas position
                'ini_y' : units of nm from plas position
                'ini_mol_orintation' : units of radians counter-clock from +x
        '''
        ## Get arguments and name them
        normed_raveled_image_data = fit_args[0]
        plas_centroid = fit_args[1]
        if len(fit_args) == 3:
            integral_normalize = fit_args[2]
        elif len(fit_args) == 2:
            integral_normalize = False

        ## Get arg to determine wheter pixelated model is calculated
        ## by averageing model across each pixel.
        if 'avg_model_over_pixels' in fit_kwargs:
            avg_model_over_pixels = fit_kwargs['avg_model_over_pixels']
        else:
            avg_model_over_pixels = False

        ## Define model image, with 'for_plot' param increasing image
        ## resolution for pixel averaging.
        raveled_model = self.raveled_model_of_params(
            fit_params,
            plas_centroid=plas_centroid,
            for_plot=avg_model_over_pixels
            )

        if avg_model_over_pixels:
            ## Average model over pixels.
            ## start by reshaping image
            image_array = raveled_model.reshape(self.plt_obs_points[-2].shape)
            ## Perform the pixel average
            # image_array = self.bin_ndarray(image_array, self.obs_points[-2].shape)
            image_array = self.rebin(image_array, self.obs_points[-2].shape)
            ## unravel again
            raveled_model = image_array.ravel()


        if integral_normalize:
            normed_raveled_model = raveled_model/(
                raveled_model.sum()
                /
                (self.sensor_size/cm_per_nm)**2. ## A in cm
                )

        elif not integral_normalize:
            normed_raveled_model = raveled_model/np.max(raveled_model)

        return normed_raveled_model - normed_raveled_image_data

    def raveled_model_of_params(self,
        fit_params,
        plas_centroid,
        for_plot=False
        ):
        """ Returns raveled model image as a function of fit parameters.
            'for_plot' uses higher res 'obs_points'.
            """
        ## Add z-dimension to molecule locations
        locations = np.array([[fit_params[0], fit_params[1], 0]])

        ## add z-dimension to plasmon location
        plas_centroid = np.array([[plas_centroid[0], plas_centroid[1], 0]])

        ## np.least_squares doesn't want to take a nested list for the
        ## 3D molecule, so here we assume that is fit_params is len = 4
        ## then elements 2 and 3 are theta and phi
        if len(fit_params) == 3:
            _angle = fit_params[2]
        elif len(fit_params) == 4:
            _angle = np.array([[fit_params[2], fit_params[3]]])
        else:
            raise ValueError("Wrong number of model parameters, must "/
             "be 3 for molecule oriented in focal plane, or 4 if 3D.")
        if not for_plot:
            obs_points = self.obs_points
        elif for_plot:
            obs_points = self.plt_obs_points

        ## Define model instance
        exp_instance = MolCoupNanoRodExp(
            locations,
            mol_angle=_angle,
            plas_centroid=plas_centroid,
            plas_angle=self.rod_angle,
            obs_points=obs_points,
            for_fit=True,
            ## List system parameters to eliminate repetative reference
            ## to .yaml during fit routine.
            drive_energy_eV=self.drive_energy_eV,
            eps_inf=self.eps_inf,
            omega_plasma=self.omega_plasma,
            gamma_drude=self.gamma_drude,
            a_long_meters=self.a_long_meters,
            a_short_meters=self.a_short_meters,
            eps_b=self.eps_b,
            fluo_quench_range=self.fluo_quench_range,
            fluo_ext_coef=self.fluo_ext_coef,
            fluo_mass_hbar_gamma=self.fluo_mass_hbar_gamma,
            fluo_nr_hbar_gamma=self.fluo_nr_hbar_gamma,
            drive_I=self.drive_I,
            sensor_size=self.sensor_size,
            is_sphere=self.is_sphere,
            drive_amp=self.drive_amp,
            sphere_model=self.sphere_model
            )

        ## Get model image
        raveled_model = exp_instance.anal_images[0].ravel()

        return raveled_model

    def plot_image_from_params(self, fit_params, plas_centroid, ax=None):
        raveled_image = self.raveled_model_of_params(
            fit_params,
            plas_centroid,
            for_plot=True
            )
        self.plot_raveled_image(raveled_image, ax)
        # plt.quiver(self.mol_locations[ith_molecule, 0], self.mol_locations[ith_molecule, 1],
                   # np.cos(self.mol_angles[ith_molecule]),np.sin(self.mol_angles[ith_molecule]),
                   # color='white',pivot='middle')

    def plot_raveled_image(self, image, ax=None):
        if image.shape[-1] == (self.plt_obs_points[-2].shape[-1])**2:
            obs_points = self.plt_obs_points
        elif image.shape[-1] == (self.obs_points[-2].shape[-1])**2:
            obs_points = self.obs_points

        if ax is None:
            plt.figure(figsize=(3,3),dpi=600)
            ## Assume plotting data
            num_pixels = obs_points[-2].shape[0]
            pixel_width_cm = (
                (obs_points[-2].max() - obs_points[-2].min())
                /
                (num_pixels - 1)
                )
            image_size_nm = pixel_width_cm * num_pixels / cm_per_nm
            plt.imshow(
                image.reshape(obs_points[-2].shape).T, ## This has to be transposed in order to match result I would get with pcolor()
                origin='lower',
                extent=[
                    -image_size_nm/2,
                    image_size_nm/2,
                    -image_size_nm/2,
                    image_size_nm/2
                    ]
                )
            plt.colorbar()
        else:
            ax.contour(obs_points[-2]/cm_per_nm,
                obs_points[-1]/cm_per_nm,
                image.reshape(obs_points[-2].shape),
                cmap='Greys',
                linewidths=0.5,
                )
        plt.title(r'$|E|^2/|E_\mathrm{inc}|^2$')
        plt.xlabel(r'$x$ [nm]')
        plt.ylabel(r'$y$ [nm]')
        return plt.gca()

    def plot_fit_results_as_quiver_map(
        self,
        fitted_exp_instance,
        plot_limits=None,
        given_ax=None,
        draw_quadrant=True,
        arrow_colors=None,
        ):
        '''...'''

        if not hasattr(self, 'model_fit_results'):
            self.fit_model_to_image_data()
        if plot_limits is None:
            plot_limits = fitted_exp_instance.default_plot_limits

        self.mol_angles = fitted_exp_instance.mol_angles

        quiv_ax, = self.quiver_plot(
            x_plot=fitted_exp_instance.mol_locations[:,0],
            y_plot=fitted_exp_instance.mol_locations[:,1],
            ## For angles, grab last col, as this will be the in plane
            ## ange regardless if model is 2d or 3d.
            angles=self.model_fit_results[:,-1],
            plot_limits=plot_limits,
            true_mol_angle = fitted_exp_instance.mol_angles,
            nanorod_angle = fitted_exp_instance.rod_angle,
            title=r'Model Fit Pol. and Loc.',
            given_ax=given_ax,
            draw_quadrant=draw_quadrant,
            arrow_colors=arrow_colors,
            )
        self.scatter_centroids_wLine(
            fitted_exp_instance.mol_locations[:,0],
            fitted_exp_instance.mol_locations[:,1],
            self.model_fit_results[:,:2].T,
            quiv_ax
            )

        return quiv_ax

    def plot_contour_fit_over_data(self, image_idx):
            ax = self.plot_raveled_image(self.image_data[image_idx])
            self.plot_image_from_params(
                self.model_fit_results[image_idx],
                plas_centroid=self.plas_centroids[image_idx],
                ax=ax)
            return ax


def fixed_ori_mol_placement(
    x_min=0,
    x_max=350,
    y_min=0,
    y_max=350,
    mol_grid_pts_1D = 10,
    mol_angle=0
    ):

    locations = diffi.observation_points(
        x_min, x_max, y_min, y_max, points=mol_grid_pts_1D
        )[0]
    locations = np.hstack((locations,np.zeros((locations.shape[0],1))))

    mol_linspace_pts = mol_grid_pts_1D
#     random_mol_angles= (np.random.random(mol_linspace_pts**2)*np.pi*2)
    return [locations, mol_angle]

def random_ori_mol_placement(
    x_min=0, x_max=350, y_min=0, y_max=350, mol_grid_pts_1D = 10):
    locations = diffi.observation_points(
        x_min, x_max, y_min, y_max, points=mol_grid_pts_1D
        )[0]
    locations = np.hstack((locations,np.zeros((locations.shape[0],1))))

    mol_linspace_pts = mol_grid_pts_1D
    random_mol_angles_0To360= (np.random.random(mol_linspace_pts**2)*np.pi*2)
    return [locations, random_mol_angles_0To360]

if __name__ == '__main__':
    '''This shit is all broken, or at least ucm_per_nmaintained'''

    print('Just sit right back while I do nothing.')
