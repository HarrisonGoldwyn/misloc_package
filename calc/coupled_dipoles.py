"""
Module for general electrodynamics of electric dipoles, their interactions, and observables.
"""

from misloc_mispol_package import project_path
from misloc_mispol_package.optics import anal_foc_diff_fields as aff

import numpy as np
import scipy.special as spl

import yaml

parameter_files_path = (
    project_path + '/param'
)

phys_const_file_name = '/physical_constants.yaml'
opened_constant_file = open(
    parameter_files_path+phys_const_file_name,
    'r')
constants = yaml.load(opened_constant_file,
                      Loader=yaml.SafeLoader) # (Zu Edit: Loader=yaml.SafeLoader)
e = constants['physical_constants']['e']
c = constants['physical_constants']['c']  # charge of electron in statcoloumbs
hbar = constants['physical_constants']['hbar']
nm = constants['physical_constants']['nm']
n_a = constants['physical_constants']['nA']


## Adopted from old oscillator code
def fluorophore_mass(ext_coef, gamma, n_b):
    '''Derived at ressonance'''
    m = 4 * np.pi * e**2 * n_a  / (
            ext_coef * np.log(10) * c * n_b * gamma
            )
    return m

## Define polarizabilities in diagonal frames
def sparse_polarizability_tensor(mass, w_res, w, gamma_nr, a, eps_inf, eps_b):
    '''Define diagonal polarizability with single cartesien component derived
        from Drude model > Clausius-mosati.
        Assumes physical constants definded; e, c
    '''
    gamma_r = gamma_nr + (2*e**2./(3*mass*c**3.))*w**2.
    alpha_0_xx_osc = (e**2. / mass)/(w_res**2. - w**2. - 1j*gamma_r*w)
    alpha_0_xx_static = (a**3. * (eps_inf - 1*eps_b)/(eps_inf + 2*eps_b))
    alpha_0_xx = alpha_0_xx_osc + alpha_0_xx_static

    if type(alpha_0_xx) is np.ndarray and alpha_0_xx.size > 1:
        alpha_0_xx = alpha_0_xx[..., None, None]

    alpha_0_ij = alpha_0_xx * np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ])

    return alpha_0_ij

def sparse_ellipsoid_polarizability(eps, eps_b, a_x, a_y, a_z):
    ''' '''
    def L_i(a, b, c):
        ''' assumes 'a' is ith radius for 'L_i'
        Don't think this will trivially converge'''
        q = np.linspace(0,10000,100000)*nm**2.
        ## these parameters led to ...
        ##... L_1 + L_2 + L_3 = ~.98 for my 44 x 20 x 20 nm rod
        ## not sure if they will
        fq = ( (a**2. + q) * (b**2. + q) * (c**2. + q) )**0.5
        integrand = 1/( (a**2. + q) * fq )
        integral = np.trapz(integrand, q)
        L_val = (a*b*c/2) * integral

        return L_val

    def alpha_ii(a, b, c):
        alpha = a*b*c * (eps - eps_b)/(
            3*eps_b + 3*L_i(a,b,c)*(eps-eps_b)
            )
        return alpha

    alpha_1 = alpha_ii(a_x,a_y,a_z)
    alpha_2 = alpha_ii(a_y,a_z,a_x)
    alpha_3 = alpha_ii(a_z,a_x,a_y)

    alpha_ij = np.array([[alpha_1,      0.,      0.],
                         [     0., alpha_2,      0.],
                         [     0.,      0., alpha_3]])
    return alpha_ij

def drude_model(w, eps_inf, w_p, gamma):
    ''' '''
    eps = eps_inf - w_p**2./(w**2. + 1j*w*gamma)
    return eps

def drude_lorentz_model(w, eps_inf, w_p, gamma, f_1, w_1):
    ''' '''
    eps = eps_inf - w_p**2. * (
        (1-f_1)/(w**2. + 1j*w*gamma)
        +
        f_1/(w**2. + 1j*w*gamma - w_1**2.)
        )
    return eps

def sparse_ellipsoid_polarizability_drude(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_y, a_z):
    ''' '''
    return sparse_ellipsoid_polarizability(
        drude_model(w, eps_inf, w_p, gamma), eps_b, a_x, a_y, a_z)

def sigma_scat_spheroid(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_y, a_z):
    ''''''
    alpha = sparse_ellipsoid_polarizability_drude(
        w, eps_inf, w_p, gamma, eps_b, a_x, a_y, a_z)

    sigma = (8*np.pi/3)*(w/c)**4.*(np.abs(alpha[0,0])**2.
        # + np.abs(alpha[1,1])**2.
        )
    return sigma


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## retarded ellipsoid from Kong's notes
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sparse_ret_prolate_spheroid_polarizability(
    eps,
    eps_b,
    a_x,
    a_yz,
    w,
    isolate_mode=None):
    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        for the prolate spheroid (a_x < a_yz).

        'isolate_mode' takes args
            'long' : just x axis for prolate sphereoid (a_x > a_yz) or
                x and y axes for oblate spheroid (a_x < a_yz)
            'short' : y and x axes for prolate sphereoid (a_x > a_yz) or
                just z axis for oblate spheroid (a_x < a_yz)

        '''
    ### Define QS polarizability 'alphaR'
    def alphaR_ii(i, a_x, a_yz):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''

        alpha = ((a_x*a_yz**2.)/3) * (eps - eps_b)/(
            eps_b + L_i(i,a_x,a_yz)*(eps-eps_b)
            )

        return alpha

    ### Define static geometric factors 'L_i' and eccentricity 'ecc' required
    ### for quasistatic alpha 'alphaR'
    def L_i(i, a_x, a_yz):
        ''' '''
        def L_x(a_x, a_yz):
            e = ecc(a_x, a_yz)

            if a_x > a_yz:
                ## Use prolate result
                L = (1-e**2.)/e**3. * (-e + np.arctanh(e))
            elif a_x < a_yz:
                ## Use oblate spheroid result
                L = (1/e**2.)*(1- (np.sqrt(1-e**2.)/e)*np.arcsin(e))

            return L

        def L_yz(a_x, a_yz):
            ''' 1 - L_x = 2*L_yx '''
            return (1 - L_x(a_x, a_yz))/2.

        if i == 1:
            L = L_x(a_x, a_yz)
        elif (i == 2) or (i == 3):
            L = L_yz(a_x, a_yz)

        return L

    def ecc(a_x, a_yz):
        return np.sqrt(np.abs(a_x**2. - a_yz**2.)/max([a_x, a_yz])**2.)

    ### Define retardation correction to alphaR
    def alphaMW_ii(i, a_x, a_yz):
        alphaR = alphaR_ii(i, a_x, a_yz)
        k = w*np.sqrt(eps_b)/c

        if i == 1:
            l_E = a_x
            D = D_x(a_x, a_yz)
        elif (i == 2) or (i == 3):
            l_E = a_yz
            D = D_yz(a_x, a_yz)

        alphaMW = alphaR/(
            1
            - (k**2./l_E) * D * alphaR
            - 1j * ((2*k**3.)/3) * alphaR
            )

        return alphaMW

    ### Define dynamic geometric factors 'D_i' for alphaMW
    def D_x(a_x, a_yz):
        e = ecc(a_x, a_yz)
        if a_x > a_yz:
            ## Use prolate result
            D = 3/4 * (
                ((1+e**2.)/(1-e**2.))*L_i(1, a_x, a_yz) + 1
                )
        elif a_x < a_yz:
            ## Use oblate result
            D = 3/4 * ((1-2*e**2.)*L_i(1, a_x, a_yz) + 1)
        return D

    def D_yz(a_x, a_yz):
        e = ecc(a_x, a_yz)

        if a_x > a_yz:
            D = (a_yz/(2*a_x))*(3/e * np.arctanh(e) - D_x(a_x,a_yz))
        elif a_x < a_yz:
            D = (a_yz/(2*a_x))*(
                3*np.sqrt(1-e**2.)/e * np.arcsin(e) - D_x(a_x,a_yz))

        return D

    if a_x > a_yz:
        ## For prolate spheroid, assign long axis to be x
        alpha_11 = alphaMW_ii(1, a_x, a_yz)
        alpha_22 = alphaMW_ii(2, a_x, a_yz)
        alpha_33 = alphaMW_ii(3, a_x, a_yz)
    elif a_x < a_yz:
        ## For oblate spheroid, assign short axis to be z
        alpha_11 = alphaMW_ii(2, a_x, a_yz)
        alpha_22 = alphaMW_ii(2, a_x, a_yz)
        alpha_33 = alphaMW_ii(1, a_x, a_yz)

    if isolate_mode == None:                # (Zu Edit: is -> ==)
        alpha_ij = np.array([[ alpha_11,       0.,       0.],
                             [       0., alpha_22,       0.],
                             [       0.,       0., alpha_33]])

    elif isolate_mode == 'long':            # (Zu Edit: is -> ==)
        if a_x > a_yz:
            ## Keep only alpha_x for prolate
            alpha_ij = np.array([
                [ alpha_11,       0.,       0.],
                [       0.,       0.,       0.],
                [       0.,       0.,       0.]
                ])
        elif a_x < a_yz:
            ## Keep alpha_x and #alpha_y for oblate
            alpha_ij = np.array([
                [ alpha_11,       0.,       0.],
                [       0., alpha_22,       0.],
                [       0.,       0.,       0.]
                ])
    elif (isolate_mode == 'short') or (isolate_mode == 'trans'):    # (Zu Edit: is -> ==)
        if a_x > a_yz:
            alpha_ij = np.array([
                [       0.,       0.,       0.],
                [       0., alpha_22,       0.],
                [       0.,       0., alpha_33]
                ])
        elif a_x < a_yz:
            alpha_ij = np.array([
                [       0.,       0.,       0.],
                [       0.,       0.,       0.],
                [       0.,       0., alpha_33]
                ])


    return alpha_ij



def sparse_ret_prolate_spheroid_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_yz, isolate_mode=None):
    ''' '''
    return sparse_ret_prolate_spheroid_polarizability(
       drude_model(w, eps_inf, w_p, gamma), eps_b, a_x, a_yz, w, isolate_mode)

# For parameterization by spectra fit or modeling spectra
def sigma_prefactor(w, eps_b):
    """ added for debugging on 02/20/19 """
    n_b = np.sqrt(eps_b)
    prefac = (
        (8*np.pi/3)*(w * n_b/ c)**4.
        /(
        # 0.5
        # *
        n_b
        ) # copied from MNPBEM source
        )
    return prefac

def long_sigma_scat_ret_pro_ellip(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_yz):
    ''''''
    alpha = sparse_ret_prolate_spheroid_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a_x, a_yz)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[0,0])**2.
        )
    return sigma

def short_sigma_scat_ret_pro_ellip(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_yz):
    ''''''
    alpha = sparse_ret_prolate_spheroid_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a_x, a_yz)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[1,1])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[2,2])**2.
        )
    return sigma


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Adapting the retarded ellipsoid section to be nonsingular for spheres
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sparse_ret_sphere_polarizability(
    eps,
    eps_b,
    a,
    w,
    isolate_mode=None,
    ):

    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        '''
    ### Define QS polarizability 'alphaR'
    def alphaR_ii(i, a):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''

        alpha = ((a**3.)/3) * (eps - eps_b)/(
            eps_b + (1/3)*(eps-eps_b)
            )

        return alpha


    ### Define retardation correction to alphaR
    def alphaMW_ii(i, a):
        alphaR = alphaR_ii(i, a)
        k = w*np.sqrt(eps_b)/c

        alphaMW = alphaR/(
            1
            -
            ((k**2./a) * alphaR)
            -
            (1j * ((2*k**3.)/3) * alphaR)
            )

        return alphaMW

    alpha_11 = alphaMW_ii(1, a)
    alpha_22 = alphaMW_ii(2, a)
    alpha_33 = alphaMW_ii(3, a)

    alpha_tensor = distribute_sphere_alpha_components_into_tensor(
        alpha_11,
        alpha_22,
        alpha_33,
        isolate_mode,
        )

    return alpha_tensor


def sparse_sphere_polarizability_TMatExp(
    eps,
    eps_b,
    a,
    w,
    isolate_mode=None,
    ):

    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        '''
    ### Define QS polarizability 'alphaR'
    def alphaTME_ii(a):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''
        eps_r = eps / eps_b
        ka = w*np.sqrt(eps_b)/c * a

        alpha = (eps_r - 1)/(
            eps_r + 2
            -
            (6*eps_r - 12)*(ka**2./10)
            -
            1j*(2*ka**3./3)*(eps_r - 1)
            ) * a**3.

        return alpha

    alpha_11 = alphaTME_ii(a)
    alpha_22 = alphaTME_ii(a)
    alpha_33 = alphaTME_ii(a)

    alpha_tensor = distribute_sphere_alpha_components_into_tensor(
        alpha_11,
        alpha_22,
        alpha_33,
        isolate_mode,
        )

    return alpha_tensor


def sparse_sphere_polarizability_Mie(
    eps,
    eps_b,
    a,
    w,
    isolate_mode=None,
    ):

    ''' Polarizability that results from the exact dipole Mie coefficient.
        '''
    def alphaTME_ii(a):
        ''' returns components of alpha in diagonal basis '''
        eps_r = eps / eps_b
        m = np.sqrt(eps_r)
        k = w*np.sqrt(eps_b)/c
        x = k*a

        j1x = spl.spherical_jn(1,x)
        xj1x_prime = (
            spl.spherical_jn(1,x) +
            x*spl.spherical_jn(1,x, derivative=True)
            )

        j1mx = spl.spherical_jn(1,m*x)
        mxj1mx_prime = (
            spl.spherical_jn(1,m*x) +
            m*x*spl.spherical_jn(1,m*x, derivative=True)
            )

        def h1(x, der):
            return (
                spl.spherical_jn(1, x, derivative=der)
                +
                1j*spl.spherical_yn(1, x, derivative=der)
                )
        h1x = h1(x, False)
        xh1x_prime = (
            h1(x, False) +
            x*h1(x, True)
            )

        a_mie =(
            (m**2.*j1mx*xj1x_prime - j1x*mxj1mx_prime)
            /
            (m**2.*j1mx*xh1x_prime - h1x*mxj1mx_prime)
            )

        alpha = 1j*3/(2*k**3.)*a_mie

        return alpha

    alpha_11 = alphaTME_ii(a)
    alpha_22 = alphaTME_ii(a)
    alpha_33 = alphaTME_ii(a)

    alpha_tensor = distribute_sphere_alpha_components_into_tensor(
        alpha_11,
        alpha_22,
        alpha_33,
        isolate_mode,
        )

    return alpha_tensor


def distribute_sphere_alpha_components_into_tensor(
    alpha_11,
    alpha_22,
    alpha_33,
    isolate_mode):

    ## Reorganize matrix dimensions if multiple frequencies given
    if type(alpha_11) is np.ndarray and alpha_11.size > 1:
        alpha_11 = alpha_11[..., None, None]
    if type(alpha_22) is np.ndarray and alpha_22.size > 1:
        alpha_22 = alpha_22[..., None, None]
    if type(alpha_33) is np.ndarray and alpha_33.size > 1:
        alpha_33 = alpha_33[..., None, None]

    if isolate_mode == None:
        alpha_ij = (
            alpha_11 * np.array([
                [1.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
                ])
            +
            alpha_22 * np.array([
                [0.,0.,0.],
                [0.,1.,0.],
                [0.,0.,0.]
                ])
            +
            alpha_33 * np.array([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,1.]
                ])
            )
    elif isolate_mode == 'long':
        alpha_ij = (
            alpha_11 * np.array([
                [1.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
                ])
            )
    elif (isolate_mode == 'short') or (isolate_mode == 'trans'):
        alpha_ij = (
            alpha_22 * np.array([
                [0.,0.,0.],
                [0.,1.,0.],
                [0.,0.,0.]
                ])
            +
            alpha_33 * np.array([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,1.]
                ])
            )

    return alpha_ij


def sparse_TMatExp_sphere_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a, isolate_mode=None):
    ''' '''
    return sparse_sphere_polarizability_TMatExp(
       drude_model(w, eps_inf, w_p, gamma),
       eps_b,
       a,
       w,
       isolate_mode,
       )


def sparse_Mie_sphere_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a, isolate_mode=None):
    ''' '''
    return sparse_sphere_polarizability_Mie(
       drude_model(w, eps_inf, w_p, gamma),
       eps_b,
       a,
       w,
       isolate_mode,
       )


def sparse_ret_sphere_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a, isolate_mode=None):
    ''' '''
    return sparse_ret_sphere_polarizability(
       drude_model(w, eps_inf, w_p, gamma),
       eps_b,
       a,
       w,
       isolate_mode,
       )


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Define scattering crossections for the 3 sphere models
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sigma_scat_ret_sphere(w, eps_inf, w_p, gamma,
    eps_b, a,):
    ''''''
    alpha = sparse_ret_sphere_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[...,0,0])**2.
        )
    return sigma

def sigma_scat_Mie_sphere(w, eps_inf, w_p, gamma,
    eps_b, a,):
    ''''''
    alpha = sparse_Mie_sphere_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[...,0,0])**2.
        )
    return sigma

def sigma_scat_TMatExp_sphere(w, eps_inf, w_p, gamma,
    eps_b, a,):
    ''''''
    alpha = sparse_TMatExp_sphere_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[...,0,0])**2.
        )
    return sigma

# def short_sigma_scat_ret_sphere(w, eps_inf, w_p, gamma,
#     eps_b, a_x, a_yz):
#     ''''''
#     alpha = sparse_ret_sphere_polarizability_Drude(
#         w, eps_inf, w_p, gamma, eps_b, a_x, a_yz)

#     ## result I had as of 02/19/19, don't remember justification
#     # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
#     #     np.abs(alpha[1,1])**2.
#     #     )

#     ## simple fix, changing k -> w*n/c
#     sigma = sigma_prefactor(w, eps_b) * (
#         np.abs(alpha[1,1])**2.
#         )
#     # print('(8*np.pi/3)*(w/c)**4. = ',(8*np.pi/3)*(w/c)**4. )
#     # print('(np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.) = ',
#         # (np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.))
#     return sigma


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Coupling stuff...
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### For generalized polarizabilities

def dipole_mags_gened(
    mol_angle,
    plas_angle,
    d_col,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=None,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, 3).

        Assumes 3D dipoles when 'mol_angle' is 2 dimensional,
        interpreted as a list of (theta, phi) coordinate pairs.
        """


    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    alpha_0, E_drive = rotate_molecule(
        mol_angle=mol_angle,
        alpha0_diag=alpha0_diag,
        E_d_angle=E_d_angle,
        drive_amp=drive_amp,
        )

    alpha_1_p1 = alpha1_diag
    alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

    G_d = G(drive_hbar_w, d_col, n_b)

    geometric_coupling_01 = np.linalg.inv(
        np.identity(3) - alpha_0 @ G_d @ alpha_1 @ G_d
        )

    p0 = np.einsum('...ij,...j->...i',geometric_coupling_01 @ alpha_0, E_drive)
    p1 = np.einsum('...ij,...j->...i',alpha_1 @ G_d, p0)

    return [p0, p1]



def plas_dip_driven_by_mol(
    mol_angle,
    plas_angle,
    d_col,
    mol_dipole_mag,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=1,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, 3).
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = mol_angle ## angle of bf_p0 in lab frame

    # Initialize unit vecotr for molecule dipole in lab frame
    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    if E_d_angle == None:
        E_d_angle = mol_angle
    # rotate driving field into lab frame
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*drive_amp

    ## Build 3D molecule dipole moments
    if mol_dipole_mag.ndim != 1:            # (Zu Edit: is not -> !=)
        raise TypeError(f"'mol_dipole_mag' is not dimension 1\n"+
            f"mol_dipole_mag.ndim = {mol_dipole_mag.ndim}")

    num_dips_for_calc = len(mol_dipole_mag)

    ## Creat diagonal polarizability for molecule
    alpha0_diag = np.zeros((num_dips_for_calc, 3, 3), dtype=np.complex_)
    alpha0_diag[..., 0, 0] = mol_dipole_mag/drive_amp
    ## Rotate molecule dipoles according to given angle
    alpha_0 = rotation_by(-phi_0) @ alpha0_diag @ rotation_by(phi_0)

    ## Rotate plasmon polarizability by given angle
    alpha_1 = rotation_by(-phi_1) @ alpha1_diag @ rotation_by(phi_1)

    ## Build coupling tensor
    G_d = G(drive_hbar_w, d_col, n_b)

    geometric_coupling_01 = np.linalg.inv(
        np.identity(3) - alpha_0 @ G_d @ alpha_1 @ G_d
        )

    p0 = np.einsum('...ij,...j->...i',alpha_0, E_drive)
    p1 = np.einsum('...ij,...j->...i',alpha_1 @ G_d, p0)

    return [p0, p1]


### older stuff in terms of effective masses and whatnot...
def rotation_by(by_angle, rot_axis='z'):
    ''' need to vectorize '''
    if type(by_angle)==np.ndarray or type(by_angle)==list:
        R = np.zeros((by_angle.size,3,3))
    else:
        R = np.zeros((3,3))
    cosines = np.cos(by_angle)
    sines = np.sin(by_angle)

    # if rot_axis == 'z':
    #     R[..., 0, 0] = cosines
    #     R[..., 0, 1] = sines
    #     R[..., 1, 0] = -sines
    #     R[..., 1, 1] = cosines
    #     R[..., 2, 2] = 1
    # elif rot_axis == 'y':
    #     R[..., 2, 2] = cosines
    #     R[..., 2, 0] = sines
    #     R[..., 0, 2] = -sines
    #     R[..., 0, 0] = cosines
    #     R[..., 1, 1] = 1
    # elif rot_axis == 'x':
    #     R[..., 1, 1] = cosines
    #     R[..., 1, 2] = sines
    #     R[..., 2, 1] = -sines
    #     R[..., 2, 2] = cosines
    #     R[..., 0, 0] = 1

    if rot_axis == 'x':
        R[..., 1, 1] = cosines
        R[..., 1, 2] = -sines
        R[..., 2, 1] = sines
        R[..., 2, 2] = cosines
        R[..., 0, 0] = 1
    elif rot_axis == 'y':
        R[..., 0, 0] = cosines
        R[..., 0, 2] = sines
        R[..., 2, 0] = -sines
        R[..., 2, 2] = cosines
        R[..., 1, 1] = 1
    elif rot_axis == 'z':
        R[..., 0, 0] = cosines
        R[..., 0, 1] = -sines
        R[..., 1, 0] = sines
        R[..., 1, 1] = cosines
        R[..., 2, 2] = 1

    return R


## define coupling diad
def G(drive_hbar_w, d_col, n_b):
    ''' Dipole relay tensor at frequency 'drive_hbar_w'/hbar, evaluated
        at point specified by vector 'd_col' assuming the source dipole \
        at origin. Background index is determined by

        Arg details:
            d_col.shape : shape = (...,3) -> interpretable as ...
                number of row vectors.
        '''

    d = vec_mag(d_col) ## returns shape = (...,1), preserves dimension
    n_hat = d_col/d ## returns shape = (...,3)

    w = drive_hbar_w/hbar
    k = w * n_b / c

    dyad = np.einsum('...i,...j->...ij',n_hat,n_hat)

    ## If 1 seperation is given, check if multable frequencies given for spectrum
    if d.size != 1:                                     # (Zu Edit: is not -> !=)
        d = d[...,None]
    elif d.size == 1 and (type(k) is np.ndarray):       # (Zu Edit: is -> ==)
        if k.size > 1:
            k = k.reshape(((k.size,)+(dyad.ndim-1)*(1,)))

    complex_phase_factor = np.exp(1j*k*d)

    ## add all piences together to calculate coupling
    g_dip_dip = (
        # normalization
        # *
        (
            complex_phase_factor*(
                (3.*dyad - np.identity(3)) * (1/d**3.- 1j*k/d**2.)
                -
                (dyad - np.identity(3)) * (k**2./d)
                )
            )
        )

    return g_dip_dip


### ^ requires
def vec_mag(row_vecs):
    ''' Replace last dimension of array with normalized verion
        '''
    vector_magnitudes = np.linalg.norm(row_vecs, axis=(-1))[:,None]  # breaks if mag == 0, ok?
    return vector_magnitudes


def eV_to_Hz(energy):
    return energy/hbar


def uncoupled_p0(
    mol_angle,
    E_d_angle=None,
    alpha_0_p0=None,
    drive_amp=None,
    return_polarizability_tensor=False,
    ):

    alpha_0, E_drive = rotate_molecule(
        mol_angle=mol_angle,
        alpha0_diag=alpha_0_p0,
        E_d_angle=E_d_angle,
        drive_amp=drive_amp)

    p0_unc = np.einsum('...ij,...j->...i', alpha_0, E_drive)

    if return_polarizability_tensor:
        return [p0_unc, alpha_0]
    else:
        return [p0_unc]


def rotate_molecule(mol_angle, alpha0_diag, E_d_angle, drive_amp):
    """ Returns rotated polarizability tensor and driving field vector
        """
    if E_d_angle == None:
        E_d_angle = mol_angle

    # rotate driving field into lab frame
    if np.asarray(E_d_angle).ndim == 2:
        ## If two dimensional, start with z oriented dipole and rotate
        ## to polar angle.
        E_d_theta = E_d_angle[:, 0] ## 1D
        E_d_phi = E_d_angle[:, 1] ## 1D

        E_drive = np.array([0, 0, 1])*drive_amp
        E_drive = rotation_by(E_d_theta, rot_axis='y') @ E_drive.T

    else:
        ## Otherize start with an x oriented dipole
        E_drive = np.array([1, 0, 0])*drive_amp
        E_d_phi = E_d_angle
    ## Perform aximuthal rotation
    E_drive = rotation_by(E_d_phi) @ E_drive.T
    ## Take out an extra dimension introduced by np.matmul
    if E_drive.shape[-1] == 1:
        E_drive = E_drive[..., 0]

    ## Build polarizability tensor for molecule
    alpha_0_p0 = alpha0_diag
    ## Decide if molecule is in focl plane or not by dimensions of
    ## the 'mol_angle' arg.
    if np.asarray(mol_angle).ndim <= 1:
        phi_0 = mol_angle ## angle of bf_p0 in lab frame
    else:
        theta_0 = mol_angle[:, 0]
        phi_0 = mol_angle[:, 1]
        ## alpha0_diag arg assumes x axis is nonzero for molecule,
        ## so first it must be rotated back to z by rotating -pi/2
        ## about y-axis.
        alpha_0_p0 = (
            rotation_by(+np.pi/2, rot_axis='y')
            @
            alpha_0_p0
            @
            rotation_by(-np.pi/2, rot_axis='y')
            )
        ## Then we can rotate it by the given polar coordinates
        alpha_0_p0 = (
            rotation_by(-theta_0, rot_axis='y')
            @
            alpha_0_p0
            @
            rotation_by(+theta_0, rot_axis='y')
            )
    ## Then rotate molecule about the aximuthal axis (by default of rotation_by().
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    return alpha_0, E_drive


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Scattering spectrum of coupled dipoles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def dipole_moments_per_omega(
    mol_angle,
    plas_angle,
    d,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag_of_omega=None,
    alpha1_diag_of_omega=None,
    n_b=None,
    drive_amp=None
    ):
    """ Computes dipole moments when only molecule dipole is driven
        """

    d_col = np.asarray(d).reshape((1, 3))

    alpha0_diag = alpha0_diag_of_omega(drive_hbar_w/hbar)
    alpha1_diag = alpha1_diag_of_omega(drive_hbar_w/hbar)

    p_0, p_1 = dipole_mags_gened(
        mol_angle,
        plas_angle,
        d_col,
        # E_d_angle=None,
        drive_hbar_w=drive_hbar_w,
        alpha0_diag=alpha0_diag,
        alpha1_diag=alpha1_diag,
        n_b=n_b,
        drive_amp=drive_amp
        )

    return p_0, p_1


def sigma_scat_coupled(
    dipoles_moments_per_omega,
    d_col,
    drive_hbar_w,
    n_b=None,
    E_0=None,
    ):
    """ Scattering spectrum of two coupled dipoles p_0 and p_1. Derived from
        Draine's prescription of the DDA.
        """

    omega = drive_hbar_w/hbar
    k = omega * n_b / c

    p_0, p_1 = dipoles_moments_per_omega(omega)

    G_d = G(drive_hbar_w, d_col, n_b)

    interference_term = np.sum((
        np.imag(p_0 * np.conj(np.einsum('...ij,...j->...i', G_d, p_1)))
        +
        np.imag(p_1 * np.conj(np.einsum('...ij,...j->...i', G_d, p_0)))
        ), axis=1)
    diag_term_0 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_0, axis=1 ))**2.
    diag_term_1 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_1, axis=1 ))**2.

    sigma = (
        (4 * np.pi * k  / np.abs(E_0)**2.)
        *
        (
            interference_term
            +
            diag_term_0
            +
            diag_term_1
            )
        )/n_b

    return [sigma, np.array(
        [interference_term, diag_term_0, diag_term_1,]
        )*(4 * np.pi * k  / (n_b*np.abs(E_0)**2.))]


def sigma_abs_coupled(
    mol_angle,
    plas_angle,
    d_col,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=None,
    ):
    """ Returns absorption crossection for two coupled dipoles with
        arguments similar to the coupled dipole formalism used
        elsewhere in this module.

        Returns: List
            List[0] : coupled absorption crossection
            List[1:3] : first dipole contribution
            List[3:5] : second dipole contribution
        """


    omega = drive_hbar_w/hbar
    k = omega * n_b / c

    p_0, p_1, alpha_0, alpha_1 = coupled_dip_mags_both_driven(
        mol_angle,
        plas_angle,
        d_col,
        E_d_angle=E_d_angle,
        drive_hbar_w=drive_hbar_w,
        alpha0_diag=alpha0_diag,
        alpha1_diag=alpha1_diag,
        n_b=n_b,
        drive_amp=drive_amp,
        return_polarizabilities=True
        )

    alpha_0_inv = np.linalg.inv(alpha_0)
    alpha_1_inv = np.linalg.inv(alpha_1)

    interference_term_0 = np.sum((
        np.imag(p_0 * np.conj(np.einsum('...ij,...j->...i', alpha_0_inv, p_0)))
        # +
        # np.imag(p_1 * np.conj(np.einsum('...ij,...j->...i', alpha_1_inv, p_1)))
        ), axis=-1)
    interference_term_1 = np.sum((
        # np.imag(p_0 * np.conj(np.einsum('...ij,...j->...i', alpha_0_inv, p_0)))
        # +
        np.imag(p_1 * np.conj(np.einsum('...ij,...j->...i', alpha_1_inv, p_1)))
        ), axis=-1)
    diag_term_0 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_0, axis=-1 ))**2.
    diag_term_1 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_1, axis=-1 ))**2.

    sigma = (
        (4 * np.pi * (k)  / np.abs(drive_amp)**2.)
        *
        (
            interference_term_0
            +
            interference_term_1
            -
            diag_term_0
            -
            diag_term_1
            )
        )

    return [sigma, np.array(
        [interference_term_0, diag_term_0, interference_term_1, diag_term_1,]
        )*(4 * np.pi * (k)  / (np.abs(drive_amp)**2.))]

def single_dip_sigma_scat(
    dipoles_moments_per_omega,
    drive_hbar_w,
    n_b=None,
    E_0=None,
    ):
    """ Scattering spectrum of two coupled dipoles p_0 and p_1. Derived from
        Draine's prescription of the DDA.
        """

    omega = drive_hbar_w/hbar
    k = omega * n_b / c

    p_0, = dipoles_moments_per_omega(omega)

    diag_term_0 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_0, axis=1 ))**2.

    sigma = (
        (4 * np.pi * k  / np.abs(E_0)**2.)
        * diag_term_0
        /n_b
        )

    return sigma


def single_dip_absorption(
    mol_angle,
    E_d_angle=None,
    alpha_0_p0=None,
    drive_hbar_w=None,
    drive_amp=None,
    n_b=None):

    ## Get dipole moment and polarizability tensor
    p, alpha = uncoupled_p0(
        mol_angle,
        E_d_angle=E_d_angle,
        alpha_0_p0=alpha_0_p0,
        drive_amp=drive_amp,
        return_polarizability_tensor=True,
        )

    alpha_inv = np.linalg.inv(alpha)

    interference_term = np.sum((
        np.imag(p * np.conj(np.einsum('...ij,...j->...i', alpha_inv, p)))
        ), axis=-1)

    omega = drive_hbar_w/hbar
    k = omega * n_b / c

    diag_term = (
        (2 / 3)
        *
        k**3
        *
        np.abs(np.linalg.norm( p, axis=-1 ))**2.
        )

    sigma = (
        (4 * np.pi * (k) / np.abs(drive_amp)**2.)
        *
        (
            interference_term
            -
            diag_term
            )
        )

    return sigma


def power_absorped(
    p,
    alpha,
    drive_hbar_w=None,
    drive_amp=None,
    n_b=None):

    # ## Get dipole moment and polarizability tensor
    # p, alpha = uncoupled_p0(
    #     mol_angle,
    #     E_d_angle=E_d_angle,
    #     alpha_0_p0=alpha_0_p0,
    #     drive_amp=drive_amp,
    #     return_polarizability_tensor=True,
    #     )

    alpha_inv = np.linalg.inv(alpha)

    interference_term = np.imag(
        np.einsum(
            '...j,...j->...',
            p,
            (np.conj(np.einsum('...ij,...j->...i', alpha_inv, p)))
            )
        )


    omega = drive_hbar_w/hbar
    k = omega * n_b / c

    diag_term = (2 / 3) * k**3 * np.abs(np.linalg.norm( p, axis=-1 ))**2.

    sigma = (
        (omega/2)
        *
        (
            interference_term
            -
            diag_term
            )
        )

    return sigma
















#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## General coupled dipoles, not neccesarily super res
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def coupled_dip_mags_both_driven(
    mol_angle,
    plas_angle,
    d_col,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=None,
    return_polarizabilities=False,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, 3).
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = mol_angle ## angle of bf_p0 in lab frame

    # Initialize unit vecotr for molecule dipole in lab frame
    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    if E_d_angle == None:
        E_d_angle = mol_angle
    # rotate driving field into lab frame
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*drive_amp


    alpha_0_p0 = alpha0_diag
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    alpha_1_p1 = alpha1_diag
    alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

    G_d = G(drive_hbar_w, d_col, n_b)

    geometric_coupling_01 = np.linalg.inv(
        np.identity(3) - alpha_0 @ G_d @ alpha_1 @ G_d
        )

    p0 = np.einsum(
        '...ij,...j->...i',
        geometric_coupling_01 @ (alpha_0 @ (np.identity(3) + G_d @ alpha_1)),
        E_drive
        )
    p1 = np.einsum(
        '...ij,...j->...i',
        geometric_coupling_01 @ (alpha_1 @ (np.identity(3) + G_d @ alpha_0)),
        E_drive
        )

    if not return_polarizabilities:
        return [p0, p1]
    ## If using to compute absorption spectrum,
    elif return_polarizabilities:
        return [p0, p1, alpha_0, alpha_1]

def single_dip_mag_pw(
    angle,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    n_b=None,
    drive_amp=None,
    return_polarizabilities=False,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, 3).
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = angle ## angle of bf_p0 in lab frame

    if E_d_angle == None:
        E_d_angle = angle
    # rotate driving field into lab frame
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*drive_amp

    alpha_0_p0 = alpha0_diag
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    p0 = np.einsum(
        '...ij,...j->...i',
        alpha_0,
        E_drive
        )

    if not return_polarizabilities:
        return [p0,]
    ## If using to compute absorption spectrum,
    elif return_polarizabilities:
        return [p0, alpha_0]


def coupled_dip_mags_focused_beam(
    mol_angle,
    plas_angle,
    d_col,
    p0_position,
    beam_x_positions,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=None,
    return_polarizabilities=False,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities driven by a focused dipole PSF beam.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, # of beam positions, 3 cart. coords).

        As with the rest of this module, 'd_col' is expected in shape
            (number of seperations, 3)

        'p0_positions' is expected in shape
            (3)

        beam_x_positions is currelty just assuming a 1d slice, so it
        should be shape
            (number of points) on the x axis.

        To be consistent with super-res notation, 'mol' dipole is
        placed at 'p0_position' and seperation vectrs point towards
        this location from the 'plas' location,
            p0_location - d_col = p1_location
        so if we want p0 on the left, d has to be negative.
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = mol_angle ## angle of bf_p0 in lab frame

    # Initialize unit vecotr for molecule dipole in lab frame
    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    k = (drive_hbar_w*n_b/hbar) / c

    ## Define positions with shape (num_seperations, 3)
    if p0_position.ndim == 1:                   # (Zu Edit: is -> ==)
        p0_position = p0_position[None, :]
    p1_position = p0_position - d_col

    ## Buld focused beam profile
    E_0 = aff.E_field(
        dipole_orientation_angle=E_d_angle,
        xi=beam_x_positions - p0_position[...,0],
        y=0,
        k=k
        ).T*drive_amp
    E_1 = aff.E_field(
        dipole_orientation_angle=E_d_angle,
        xi=beam_x_positions - p1_position[...,0],
        y=0,
        k=k
        ).T*drive_amp

    ## Normalize fields to correct beam intensity
    spot_size = 2*np.pi/k
    spot_space = np.linspace(-spot_size, spot_size, 500)
    spot_mesh = np.meshgrid(spot_space, spot_space)
    focal_spot_field = aff.E_field(
        dipole_orientation_angle=E_d_angle,
        xi=spot_mesh[0],
        y=spot_mesh[1],
        k=k
        ).T
    intensity_ofx = c/(8*np.pi) * np.sum(
        focal_spot_field*np.conj(focal_spot_field), axis=-1)

    area_image = (spot_space.max() - spot_space.min())**2.
    num_pixels = len(spot_space)**2.
    area_per_pixel = area_image / num_pixels

    beam_power = np.sum(intensity_ofx)*area_per_pixel
    ## integral of (c/8pi)|E|^2 dA = beam_power
    E_0 /= (beam_power)**0.5
    E_1 /= (beam_power)**0.5

    ## Rotate polarizabilities into connecting vector frame
    alpha_0_p0 = alpha0_diag
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    alpha_1_p1 = alpha1_diag
    alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

    G_d = G(drive_hbar_w, d_col, n_b)

    geometric_coupling_01 = np.linalg.inv(
        np.identity(3) - alpha_0 @ G_d @ alpha_1 @ G_d
        )

    p0 = np.einsum(
        '...ij,...j->...i',
        geometric_coupling_01 @ alpha_0 @ (
            np.identity(3) + G_d @ alpha_1
            ),
        E_0
        )
    p1 = np.einsum(
        '...ij,...j->...i',
        geometric_coupling_01 @ alpha_1 @ (
            np.identity(3) + G_d @ alpha_0
            ),
        E_0
        )

    if not return_polarizabilities:
        return [p0, p1]
    ## If using to compute absorption spectrum,
    elif return_polarizabilities:
        return [p0, p1, alpha_0, alpha_1]

def single_dip_mag_focused_beam(
    angle,
    p0_position,
    beam_x_positions,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    n_b=None,
    drive_amp=None,
    return_polarizabilities=False,
    ):
    """ Calculate dipole magnitude with generalized dyadic
        polarizabilities driven by a focused dipole PSF beam.

        Returns dipole moment vector as rows in array of shape
        (# of seperations, # of beam positions, 3 cart. coords).

        'p0_positions' is expected in shape
            (3)

        beam_x_positions is currelty just assuming a 1d slice, so it
        should be shape
            (number of points) on the x axis.

        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = angle ## angle of bf_p0 in lab frame


    k = (drive_hbar_w*n_b/hbar) / c

    ## Define positions with shape (num_seperations, 3)
    if p0_position.ndim == 1:                   # (Zu Edit: is -> ==)
        p0_position = p0_position[None, :]

    ## Buld focused beam profile
    E_0 = aff.E_field(
        dipole_orientation_angle=E_d_angle,
        xi=beam_x_positions - p0_position[...,0],
        y=0,
        k=k
        ).T*drive_amp

    ## Normalize fields to correct beam intensity
    spot_size = 2*np.pi/k
    spot_space = np.linspace(-spot_size, spot_size, 500)
    spot_mesh = np.meshgrid(spot_space, spot_space)
    focal_spot_field = aff.E_field(
        dipole_orientation_angle=E_d_angle,
        xi=spot_mesh[0],
        y=spot_mesh[1],
        k=k
        ).T
    intensity_ofx = c/(8*np.pi) * np.sum(
        focal_spot_field*np.conj(focal_spot_field), axis=-1)

    area_image = (spot_space.max() - spot_space.min())**2.
    num_pixels = len(spot_space)**2.
    area_per_pixel = area_image / num_pixels

    beam_power = np.sum(intensity_ofx)*area_per_pixel
    ## integral of (c/8pi)|E|^2 dA = beam_power
    E_0 /= (beam_power)**0.5

    ## Rotate polarizabilities into connecting vector frame
    alpha_0_p0 = alpha0_diag
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    ## Dipole mmoment
    p0 = np.einsum(
        '...ij,...j->...i',
        alpha_0,
        E_0
        )

    if not return_polarizabilities:
        return [p0]
    ## If using to compute absorption spectrum,
    elif return_polarizabilities:
        return [p0, alpha_0]


def partial_scattering(
    max_angle,
    p1_of_w,
    x1,
    hbar_w,
    n_b,
    E_0,
    num_int_points=1000,
    focal_l=1 #cm
    ):
    """ Returns fractional scattering crossection through aperature.
        Args:
            max_angle (float, radians): Maximum polar angle defining aperature dimension
                and practically the bounds on the integral of the
                Poynting vector.
            p1_of_w (function): Takes frequency arg and returns array
                of complex dipole magnitudes. shape = (..., 3)
            x1: position of dipole
            hbar_w: drive energy
            n_b: background index
            E_0: beam field magnitude
        """
    ## Define points on spheriacl section to evaluate fields
    sph_coord_field_points = fib.fib_alg_k_filter(
        num_points=num_int_points,
        max_ang=max_angle
        )
    # Convert spherical coordinates to Caresian.
    cart_points_on_sph = fib.sphere_to_cart(
        sph_coord_field_points[:,0],
        sph_coord_field_points[:,1],
        focal_l*np.ones(np.shape(sph_coord_field_points[:,0]))
        )
    ## Define dipole fields
    # E = G @ p
    # H = ???
    # S = E x H^*

    ## Integrate




if __name__ == "__main__":

     print("This module is not meant to be executed")




