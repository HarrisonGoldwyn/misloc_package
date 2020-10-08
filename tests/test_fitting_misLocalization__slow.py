import os
import sys

import numpy as np

## For fitting
import scipy.optimize as opt

## Load custom package modules
from ..calc import BEM_simulation_wrapper as bem
from ..calc import fitting_misLocalization as fit
from ..calc import coupled_dipoles as cp
from ..optics import diffraction_int as diffi
from ..optics import fibonacci as fib


sim_inst_JC = bem.LoadedSimExp(
        'sim_inst_JC',
        param_file='disk_JC'
        )

def test_2D_molecule_fit_with_presimulated_data():
    """ Get presimulated data from the following code:
        sim_inst_JC = bem.SimulatedExperiment(
            locations=locations,
            mol_angle=0,
            param_file='disk_JC',
            simulation_file_name='CurlyDiskJC_NoSub_dipDrive_E',
            auto_quench=False
            )
        sim_inst_JC.trial_images = sim_inst_JC.calculate_BEM_fields()
    """
    fit_inst_JC = fit.FitModelToData(
        sim_inst_JC.BEM_images,
    #     ini_guess=simTestInst_few_mol.mol_locations,
        ini_guess='gauss',
        param_file='disk_JC'
        )
    the_fit = fit_inst_JC.fit_model_to_image_data(
        check_ini=False)
    ## Check that all fit parameters are numbers
    assert np.all(~np.isnan(the_fit))

def test_3D_molecule_fit_with_presimulated_data():
    """ Data is 2D, because I have not implemented 3D molecules into
        the simulation routine at time of writing this test.
        """
    fit_inst_JC = fit.FitModelToData(
        sim_inst_JC.BEM_images,
    #     ini_guess=simTestInst_few_mol.mol_locations,
        ini_guess='gauss',
        param_file='disk_JC'
        )
    the_fit = fit_inst_JC.fit_model_to_image_data(
        let_mol_ori_out_of_plane=True,
        check_ini=False)
    ## Check that all fit parameters are numbers
    assert np.all(~np.isnan(the_fit))
