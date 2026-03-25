"""
From given galaxies, read particles and generate Lens Class
Revolves around the SubLensPart class, plus several helper functions

-> restructured to be "subdominant" w.r.t. LensModel
     -> no additional lens profiles possible (at this level)
     -> still compute lensing effects given z_lens and z_source, but those will be corrected for in the Lenstronomy profile
"""

import dill
import warnings
import numpy as np
from pathlib import Path
from functools import cached_property 

import astropy.units as u
from lenstronomy.Util.util import array2image,make_grid
#from lenstronomy.ImSim.Numerics.grid import RegularGrid

# My libs
from python_tools.tools import mkdir,to_dimless,ensure_unit
# cosmol. params.
from nazgul.lib_cosmo import SigCrit
# Get particle from galaxy catalogue
from nazgul.Translator.translator import LoadGal
# particle lens class and params.
from nazgul.particle_lenses import PartLens 
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
# likelihood class
from nazgul.likelihood import Likelihood
# project galaxy along various axis
from nazgul.project_gal import get_2Dkappa_map,ProjGal
from nazgul.project_gal import Gal2kw_samples
from nazgul.pathfinder import get_lens_lowdir_from_galdir

from nazgul.mount_doom.cracks_of_doom import BasicLensPart
from nazgul.mount_doom.cracks_of_doom import kw_prior_z_source_stnd
from nazgul.mount_doom.cracks_of_doom import pixel_num,min_thetaE,get_extents

class SubLensPart(BasicLensPart): 
    _large_attributes = ["kwargs_lens","lens_prof","Gal","PartLens","cosmo"]
    
    def __init__(self,
                 Galaxy,      # class instance of PartGal
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # number of pixels 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE, # minimum theta observable
                 subdir="./",           # subdirectory (to differentiate btw versions)
                 reload=True # reload previous instance
                 ):
        super().__init__(Galaxy=Galaxy,
                         kwlens_part=kwlens_part,
                         pixel_num=pixel_num,
                         kw_prior_z_source=kw_prior_z_source,
                         min_thetaE=min_thetaE,
                         subdir=subdir,
                         reload=reload)
        self.savedir  = get_lens_lowdir_from_galdir(galdir=self.Gal.gal_dir)
        mkdir(self.savedir)

    ### Class Structure ####
    ########################
    def _identity(self):
        """Return an immutable tuple uniquely identifying this galaxy.

        The identity is used for hashing, equality, and cache keys.
        """
        return (
            self.Gal._identity(),
            self.PartLens._identity(),
            self.pixel_num,
            self.z_source_max,
            self.kwlens_part,
            )
    
    ########################
    @property
    def name(self):
        # define name and path of savefile
        return f"Sub_{self.Gal_name}_Npix{self.pixel_num}_Part{self.PartLens_name}"
    #############################
    def run(self,read_prev=True,verbose=True):
        """Main function that computes the deflection map:
            - read the particles
            - iteratively test projections given a source at redshift 
            within z_source_min=z_lens and z_source_max(=4 by default)
                - for each projection, compute density map with AMR
                - if maximum density supercritical, select that projection
                    - sample the source redshift (ATM uniformily)
                    - find the coordinate of Maximum Density (MD) and recenter around it
                    - compute estimate of theta_E
                - else, move to next projection
                - if no projection is supercritical, discard galaxy as a lens
            - compute sigma critical
            - define grid aperture radius = 2*theta_E
                - from it and the pixel number, obtain pixel size in arcsec
            - compute the deflection map within the grid
            - create the lensed image given the source and simulated observation conditions
            - store the results
        """
        self.unpack()
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            # Lens Verification:
            ####################
            # project and check if it is a lens
            self.galaxy_projection(verbose=verbose)

            # Lensing computations:
            #######################
            self.create_lens(verbose=verbose)
            
    def create_lens(self,verbose=True):
        # setup lensing keywords
        self.PartLens.setup(self) # only run now as it needs z_source 
        # Define the radius based on ~ theta_E
        scale_tE       = 2 
        self.radius    = self.thetaE*scale_tE
        if verbose:
            print("Image radius:",np.round(self.radius,3))
        # setup lenses 
        self.setup_lenses()
        # compute all lensing maps 
        # (most computationally intense functions):
        self.alpha_map
        self.kappa_map
        self.hessian
        self.psi_map
        # Store the results
        self.store()

    # the following is meant to be rerun every time we load the class to save space
    # -> computationally not intense 
    def setup_lenses(self):
        """
        Setup lensing parameters tailored for lenstronomy
        """
        print("Setting up lensing parameters...")
        # Convert x,y,z in samples and get masses
        kw_samples = Gal2kw_samples(Gal=self.Gal,proj_index=self.proj_index,
                                    MD_coords=self.MD_coords,arcXkpc=self.arcXkpc)
        samples    = kw_samples["RAs"],kw_samples["DECs"]
        Ms         = kw_samples["Ms"]
        # Convert in lenses parameters 
        kwLnsPart,LnsProfPart  = self.PartLens.get_lens_PART(samples=samples,Ms=Ms)
        self.kwargs_lens       = kwLnsPart
        self.lens_prof         = LnsProfPart
        print("... Lensing params set up")
        
        
    def _psi_map(self,_radec=None):
        print("Computing particles lensing potential...")
        self.unpack()
        if _radec is None:
            _radec = self._radec #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_prof.function(_ra, _dec, **self.kwargs_lens)
        psi = array2image(psi)
        return psi
        
    def _alpha_map(self,_radec=None):
        print("Computing particles lensing deflection...")
        self.unpack()
        if _radec is None:
            _radec = self._radec #arcsecs  
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_prof.derivatives(_ra, _dec, **self.kwargs_lens)
        alpha_x,alpha_y = array2image(alpha_x),array2image(alpha_y)
        return alpha_x,alpha_y
        
    def _kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        print("Computing kappa map from density map...")
        if _radec is None:
            kw_extents = self.kw_extents
        else:
            kw_extents = get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        kappa = get_2Dkappa_map(Gal=self.Gal,proj_index=self.proj_index,
                                MD_coords=self.MD_coords,kwargs_extents=kw_extents,
                                SigCrit=self.SigCrit,arcXkpc=self.arcXkpc)
        return kappa
        
    def _hessian(self,_radec=None):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        """
        print("Computing hessian as gradient of the deflection map...")
        self.unpack()
        # Can be now computed also beyond the cutout
        if _radec is None:
            _radec          = self._radec  
            alpha_x,alpha_y = self.alpha_map
        else:
            alpha_x,alpha_y = self._alpha_map(_radec=_radec)
        _ra,_dec = _radec
        RA0,DEC0 = array2image(_ra)[0],array2image(_dec)[:,0]
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, RA0,DEC0)
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, RA0,DEC0)
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy
