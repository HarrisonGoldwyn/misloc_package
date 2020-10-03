from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.special as spf



def phi(x, y):
    # return np.arctan(y, x)
    return np.arctan2(-y,-x) + np.pi

def rho(x, y):
    return ( x**2. + y**2. )**0.5

def E_field(dipole_orientation_angle, xi, y, k):
    """ Defines the analytics approximation to the focused and
        diffracted field for dipole oriented in the
        focal plane at an angle 'dipole_orientation_angle' from the
        x-axis.
        """

    psi = dipole_orientation_angle
    phi_P = phi(xi, y) - psi

    ## Define Bessel ratios with limits explicitly to ovoid divergent division
    sphj1_on_krho = spf.spherical_jn( 1, k*rho(xi, y) )/(k*rho(xi, y))
    sphj1_on_krho[k*rho(xi, y) == 0] = 1/3
    ##
    j2_on_krho = spf.jv(2, k*rho(xi, y) )/(k*rho(xi, y))
    j2_on_krho[k*rho(xi, y) == 0] = 0

    E_xP = (
            (
                np.cos( phi_P )**2.
                +
                np.cos( 2*(phi_P) )
                )
            *
            sphj1_on_krho
        +
        (
            np.sin( phi_P )**2.
            *
            spf.spherical_jn( 0, k*rho(xi, y) )
            )
        )

    E_yP = (
        np.sin(phi_P)
        *
        np.cos(phi_P)
        *
        spf.spherical_jn( 2, k*rho(xi, y) )
        )

    # print("rho(xi, y) = ",rho(xi, y))
    # print("xi, y = ",xi, ' ',y)
    E_zP = -np.cos(phi_P) * j2_on_krho


    E_x = np.cos(psi)*E_xP - np.sin(psi)*E_yP
    E_y = np.sin(psi)*E_xP + np.cos(psi)*E_yP
    E_z = E_zP

    return np.array([E_x, E_y, E_z])*k**3.


def E_pz(xi, y, k):
    """ Defines the analytics approximation to the focused and
        diffracted field for dipole oriented along the optical axis.
        """
    phi_P = phi(xi, y)

    j2_on_krho = spf.jv(2, k*rho(xi, y) )/(k*rho(xi, y))
    j2_on_krho[k*rho(xi, y) == 0] = 0

    sphj0_on_krhosqrd = spf.spherical_jn(0, k*rho(xi, y) )/(k*rho(xi, y))**2.
    sphy1_plus_sphj0_on_krhosqrd = (
        spf.spherical_yn(1, k*rho(xi, y) )
        +
        sphj0_on_krhosqrd
        )
    sphy1_plus_sphj0_on_krhosqrd[k*rho(xi, y) == 0] = -2/3


    E_x = 1j*(
        j2_on_krho
        *
        np.cos(phi_P)
        )

    E_y = -1j*(
        j2_on_krho
        *
        np.sin(phi_P)
        )

    E_z = -sphy1_plus_sphj0_on_krhosqrd

    return np.array([E_x, E_y, E_z])*k**3.


def old_E_pz(xi, y, k):
    """ Defines the analytics approximation to the focused and
        diffracted field for dipole oriented along the optical axis.
        """
    phi_P = phi(xi, y)

    j2_on_krho = spf.jv(2, k*rho(xi, y) )/(k*rho(xi, y))
    j2_on_krho[k*rho(xi, y) == 0] = 0

    sphj0_on_krhosqrd = spf.spherical_jn(0, k*rho(xi, y) )/(k*rho(xi, y))**2.
    sphy1_plus_sphj0_on_krhosqrd = (
        spf.spherical_yn(1, k*rho(xi, y) )
        +
        sphj0_on_krhosqrd
        )
    sphy1_plus_sphj0_on_krhosqrd[k*rho(xi, y) == 0] = -2/3


    E_x = (
        j2_on_krho
        *
        np.cos(phi_P)
        )

    E_y = -(
        j2_on_krho
        *
        np.sin(phi_P)
        )

    E_z = 2/3 * spf.spherical_jn(0, k*rho(xi, y)) - 1/3*spf.spherical_jn(2, k*rho(xi, y))

    return np.array([E_x, E_y, E_z])*k**3.

