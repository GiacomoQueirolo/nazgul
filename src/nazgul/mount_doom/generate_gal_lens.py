"""
From given galaxies, read particles and generate Lens Class
Revolves around the GalLens class, plus several helper functions

-> restructured to be "subdominant" w.r.t. LensModel
     -> no additional lens profiles possible (at this level)
     -> still compute lensing effects given z_lens and z_source, but those will be corrected for in the Lenstronomy profile

"""

import dill
import warnings
import numpy as np
from glob import glob
from pathlib import Path
from time import gmtime, strftime
from functools import cached_property 

import astropy.units as u
from lenstronomy.Util.util import array2image,make_grid
#from lenstronomy.ImSim.Numerics.grid import RegularGrid

# My libs
from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,to_dimless,ensure_unit
# cosmol. params.
from nazgul.lib_cosmo import SigCrit
# Get all simulated particle galaxies
from nazgul.Translator import std_simsuite,std_sim
from nazgul.Translator.translator import get_all_PG,get_rnd_PG
# particle lens class and params.
from nazgul.particle_lenses import PartLens 
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
# likelihood class
from nazgul.likelihood import Likelihood
# project galaxy along various axis
from nazgul.project_gal import get_2Dkappa_map,ProjectionError
from nazgul.project_gal import Gal2kw_samples,ProjGal
from nazgul.pathfinder import get_lens_lowdir_from_galdir,get_proj_dir_from_galdir

from nazgul.mount_doom.cracks_of_doom import BasicLensPart,get_extents,kw_prior_z_source_stnd
from nazgul.configurations import scale_tE,pixel_num,min_thetaE

class GalLens(BasicLensPart): 
    _large_attributes_setup  = ["kwargs_lens","lens_prof","Gal","PartLens","cosmo"]
    _large_attributes_unpack = ["Gal","PartLens","cosmo"]
    
    def __init__(self,
                 Galaxy,      # class instance of PartGal
                 projection_index, # projection index of the galaxy
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # number of pixels 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE, # minimum theta observable
                 #subdir="./",           # subdirectory (to differentiate btw versions)
                 reload=True # reload previous instance
                 ):
        super().__init__(Galaxy=Galaxy,
                         projection_index=projection_index,
                         kwlens_part=kwlens_part,
                         pixel_num=pixel_num,
                         kw_prior_z_source=kw_prior_z_source,
                         min_thetaE=min_thetaE,
                         #subdir=subdir,
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
        name =  f"Sub_{super().name}"
        # to change asap
        return name
        
    #############################
    def run(self,read_prev=True,
            verbose=True,scale_tE=scale_tE):
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
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            self.setup()
            # Lens Verification:
            ####################
            # project and check if it is a lens
            self.galaxy_projection(verbose=verbose)

            # Lens instatiation
            ####################
            self.setup_lenses()

            # Lensing computations:
            #######################
            self.create_lens(verbose=verbose,scale_tE=scale_tE)

    def create_lens(self,verbose=True,scale_tE=scale_tE):
        # setup lensing keywords
        self.PartLens.setup(self) # only run now as it needs z_source 
        # Define the radius based on ~ theta_E
        self.radius = self.thetaE*scale_tE
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
        if not hasattr(self,"kwargs_lens") and not hasattr(self,"lens_prof"):
            print("Setting up lensing parameters... (GalLens)")
            # Convert x,y,z in samples and get masses
            self._unpack_Gal()
            self.Gal.run()
            kw_samples = Gal2kw_samples(Gal=self.Gal,proj_index=self.proj_index,
                                        MD_coords=self.MD_coords,arcXkpc=self.arcXkpc)
            samples    = kw_samples["RAs"],kw_samples["DECs"]
            Ms         = kw_samples["Ms"]
            # Convert in lenses parameters 
            self._unpack_PartLens()
            kwLnsPart,LnsProfPart  = self.PartLens.get_lens_PART(samples=samples,Ms=Ms)
            self.kwargs_lens       = kwLnsPart
            self.lens_prof         = LnsProfPart
            print("... Lensing params set up (GalLens)")
        
    def _psi_map(self,_radec=None):
        self.setup()
        if _radec is None:
            _radec = self._radec #arcsecs  
        if np.all(np.array(_radec)==np.array(self._radec)):
            if "psi_map" in self.__dict__:
                return self.__dict__["psi_map"]
                
        print("Computing particles lensing potential...")
        _ra,_dec = _radec
        self.setup_lenses()
        psi = self.lens_prof.function(_ra, _dec, **self.kwargs_lens)
        psi = array2image(psi)
        return psi
        
    def _alpha_map(self,_radec=None):
        self.setup()
        if _radec is None:
            _radec = self._radec

        if np.all(np.array(_radec) == np.array(self._radec)):
            # the following leads to recursion error:
            #if hasattr(self,"alpha_map"):
            #return self.alpha_map
            if "alpha_map" in self.__dict__:
                return self.__dict__["alpha_map"]
        print("Computing particles lensing deflection...")
        _ra,_dec = _radec
        self.setup_lenses()
        alpha_x,alpha_y = self.lens_prof.derivatives(_ra, _dec, **self.kwargs_lens)
        alpha_x,alpha_y = array2image(alpha_x),array2image(alpha_y)
        return alpha_x,alpha_y
        
    def _kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        if _radec is None:
            _radec     = self._radec
            kw_extents = self.kw_extents
        else:
            kw_extents = get_extents(arcXkpc=self.arcXkpc,_radec=_radec)

        if np.all(np.array(_radec) == np.array(self._radec)):
            if "kappa_map" in self.__dict__:
                return self.__dict__["kappa_map"]

        print("Computing kappa map from density map...")
        self._unpack_Gal()
        self.Gal.run()
        kappa = get_2Dkappa_map(Gal=self.Gal,proj_index=self.proj_index,
                                MD_coords=self.MD_coords,kwargs_extents=kw_extents,
                                SigCrit=self.SigCrit,arcXkpc=self.arcXkpc)
        return kappa
        
    def _hessian(self,_radec=None):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        """
        # Can be now computed also beyond the cutout
        if _radec is None:
            _radec          = self._radec  
            alpha_x,alpha_y = self.alpha_map
        else:
            alpha_x,alpha_y = self._alpha_map(_radec=_radec)

        if np.all(np.array(_radec) == np.array(self._radec)):
            if "hessian" in self.__dict__:
                return self.__dict__["hessian"]

        print("Computing hessian as gradient of the deflection map...")
        _ra,_dec = _radec
        RA0,DEC0 = array2image(_ra)[0],array2image(_dec)[:,0]
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, RA0,DEC0)
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, RA0,DEC0)
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy


def get_kw_galpart(kw_galpart={}):
    default_kw_galpart={"simsuite":std_simsuite,
                       "sim":std_sim}
    default_kw_galpart.update(kw_galpart)
    kw_galpart = default_kw_galpart
    return kw_galpart
    
def get_kw_lenspart(reload,kw_lenspart={}):
    default_kw_lenspart={"kwlens_part":kwlens_part_AS,
                         "projection_index":0,
                         "kw_prior_z_source":kw_prior_z_source_stnd,
                         "pixel_num":pixel_num,
                         "min_thetaE":min_thetaE,
                         "reload":reload}
    default_kw_lenspart.update(kw_lenspart)
    kw_lenspart = default_kw_lenspart
    return kw_lenspart

def gal_already_computed(Gal):
    gal_computed = False
    prj_dir = get_proj_dir_from_galdir(Gal.gal_dir)
    fls_prj = glob(f"{prj_dir}/projection_*.pkl")
    if len(fls_prj)==3:
        for fl_prj in fls_prj:
            try:
                load_whatever(fl_prj)
                gal_computed = True
            except:
                # Failed to load - something is off, need to recompute
                gal_computed = False
                return gal_computed
    return gal_computed
    
def wrapper_get_all_lens(reload=True,
                        kw_lenspart={},
                        kw_galpart={},
                        verbose=True,
                        _test=False,
                        _list_of_skippable_gals=None):
    """Get a lens from all available galaxies"""
    kw_lenspart = get_kw_lenspart(reload,kw_lenspart)
    kw_galpart  = get_kw_galpart(kw_galpart)
    all_Gal     = get_all_PG(**kw_galpart)
    all_lenses  = []
    if verbose:
        print(f"Found n={len(all_Gal)} Galaxies")
    for Gal in all_Gal:
        # Verify if all proj. have not already been computed 
        # (not important if it is a lens or not)
        print(f"\nNew Gal:\n     {Gal.name}\n")
        
        if gal_already_computed(Gal):
            print("Galaxy already computed")
            if _list_of_skippable_gals is not None:
                if Gal.name in _list_of_skippable_gals:
                    print("Skipping because in skippable list")
                    continue
            if reload:
                print("Reloading")
            else:
                print("Recomputing")
                Gal.run(reload=reload)
        else:
            Gal.run(reload=reload)
        strikes = 0
        while kw_lenspart["projection_index"]<3:
            try:
                mod_LP = GalLens(Galaxy=Gal,
                              **kw_lenspart)
                mod_LP.run(read_prev=reload)
                all_lenses.append(mod_LP)
                pji = kw_lenspart["projection_index"]
                print(f"Projection {pji} of {Gal.name} is supercritical!\n")
            except ProjectionError as PE:
                strikes+=1
            kw_lenspart["projection_index"]+=1
        if strikes==3:
            print(f"All projections of Galaxy {Gal.name} are not supercritical")
        print("Next galaxy.")
        if verbose:
            print("#########\nTime stamp:\n")
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            print("#########\n")

        kw_lenspart["projection_index"] = 0
        if _test:
            print("TEST - Stopping after only one")
            return all_lenses
    if _test:
        print("TEST - Stopping after only one")
        return all_lenses
    if verbose:
        print(f"Found n={len(all_lenses)} Lenses")
        print(f"i.e. {np.round(len(all_lenses)/len(all_Gal*3)*100,1)}% of Galaxies (considering their rotations)")
        
    return all_lenses

# get a lens no matter what:
def wrapper_get_rnd_lens(reload=True,
                        kw_lenspart={},
                        kw_galpart={}):
    """Try to get a lens from random galaxies, repeat until finds one
    which is an actual lens (i.e. supercritical)
    """    
    kw_lenspart = get_kw_lenspart(reload,kw_lenspart)
    kw_galpart  = get_kw_galpart(kw_galpart)
    while True:
        # TODO: this is efficient the first time, but if the galaxy is already computed as a lens,
        # it's a waste of time : verify if all projection are not already computed/ lensed given
        # the parameters
        Gal = get_rnd_PG(**kw_galpart)
        Gal.run()
        while kw_lenspart["projection_index"]<3:
            try:
                mod_LP = GalLens(Galaxy=Gal,
                              **kw_lenspart)
                mod_LP.run(read_prev=reload)
                return mod_LP            
            except ProjectionError as PE:
                kw_lenspart["projection_index"]+=1
        print("All projections of this galaxy are not supercritical #\n","Trying different galaxy")
        kw_lenspart["projection_index"] = 0
