"""
From given galaxies, read particles and generate Lens Class
Revolves around the SubLensPart class, plus several helper functions

-> restructured to be "subdominant" w.r.t. LensModel
     -> no additional lens profiles possible (at this level)
     -> still compute lensing effects given z_lens and z_source, but those will be corrected for in the Lenstronomy profile
"""

import dill
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
from nazgul.particle_galaxy import Gal2kwMXYZ,LoadGal
# particle lens class and params.
from nazgul.particle_lenses import PartLens 
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
# likelihood class
from nazgul.likelihood import Likelihood
# project galaxy along various axis
from nazgul.project_gal import get_2Dkappa_map,ProjGal,projection_main_AMR
from nazgul.project_gal import Gal2kw_samples
from nazgul.pathfinder import get_lens_lowdir_from_galdir

from nazgul.mount_doom.cracks_of_doom import pixel_num,min_thetaE
from nazgul.mount_doom.cracks_of_doom import kw_prior_z_source_stnd
from nazgul.mount_doom.cracks_of_doom import ReadLens
import nazgul.mount_doom.cracks_of_doom as cod

verbose       = True


class SubLensPart(): 
    def __init__(self,
                 Galaxy,      # class instance of PartGal
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # number of pixels 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE, # minimum theta observable
                 subdir="./",           # subdirectory (to differentiate btw versions)
                 reload=True # reload previous instance
                 ):

        # Wrapper class of PartGal to extend for projections
        Galaxy             = ProjGal(Galaxy) 
        # setup of data
        self.Gal           = Galaxy
        self.Gal_path      = Galaxy.pkl_path
        self.Gal_name      = Galaxy.Name # must be stored
        z_source_max       = kw_prior_z_source["z_source_max"]
        # if reload, check if Gal is a lens - if it isn't, raise error
        if reload:
            if not self.Gal.is_lens(z_source_max=z_source_max,
                                   min_thetaE=min_thetaE):
                raise RuntimeError(f"Previously defined as not a lens given z_s,max={z_source_max} and min_thetaE={min_thetaE}")

        self.savedir  = get_lens_lowdir_from_galdir(galdir=self.Gal.gal_dir)
        self.reload   = reload
        mkdir(self.savedir)
        ######
        # lensing params
        self.pixel_num     = pixel_num      
        self.kwlens_part   = kwlens_part
        self.PartLens      = PartLens(kwlens_part)
        self.PartLens_name = self.PartLens.name
        # cosmo params
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # criteria for supercriticality of the lens
        self.z_source_max  = z_source_max
        self.kw_like_zs    = cod.kw_prior2like_zs(kw_prior_z_source=kw_prior_z_source,
                                              z_lens=self.z_lens)
        self.min_thetaE    = ensure_unit(min_thetaE,u.arcsec) #arcsec

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
    
    def __hash__(self):
        """ Simplified hash based exclusively on the immutable identity tuple.
        """
        return hash(self._identity())

    def __eq__(self, other):
        """LensPart instances are equal if and only if they share
        the same conceptual identity.
        """
        if not isinstance(other, SubLensPart):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self): 
        """Human-readable identifier.

        Lazily initializes the name if it has not been generated yet.
        """
        return self.name

    # ------------------------------------------------------------------
    # Pickling support (dill / pickle)
    # ------------------------------------------------------------------

    # the following struct. is more clear and allow a slimmer stored class
    def __getstate__(self):
        """Return a slimmed-down state dictionary for serialization.

        Reconstruction is handled by `unpack()`.
        """
        state = self.__dict__.copy()
        # Large / recomputable lensing structures
        state.pop('kwargs_lens', None)
        state.pop('lens_prof', None)
        state.pop('grid',None)
        # These are reloaded / reconstructed
        state.pop('Gal',None)
        state.pop('PartLens',None)
        state.pop('cosmo',None)
        return state

    def __setstate__(self, state):
        """Restore object state from a serialized dictionary.

        NOTE:
        - This does *not* automatically reconstruct heavy attributes.
        - Lazy reconstruction is deferred until explicitly needed.
        """
        self.__dict__.update(state)
 
    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------
    def _needs_unpacking(self):
        """Check whether the object is missing reconstructed attributes.
        """
        return not all(
            hasattr(self, attr)
            for attr in ("Gal", "PartLens", "cosmo")
        )
    def _unpack(self):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking class...")
        # reload Galaxy and cosmology
        Galaxy   = LoadGal(self.Gal_path)
        Galaxy   = ProjGal(Galaxy)
        self.Gal = Galaxy
        # verify that we load the correct galaxy
        assert self.Gal_name == Galaxy.Name
        self.cosmo = Galaxy.cosmo
        
        # re-define PartLens
        self.PartLens = PartLens(self.kwlens_part)
        self.PartLens.setup(self)
        
        # Rebuild lens model if missing
        if not hasattr(self, "lens_prof"):
            self.setup_lenses()

    def unpack(self):
        """Public wrapper for lazy reconstruction.
        """
        if self._needs_unpacking():
            self._unpack()
        return self
        
    def store(self):
        """Serialize the current object to disk using dill.
        """
        with open(self.pkl_path, "wb") as f:
            dill.dump(self, f)
        print(f"Saved {self.pkl_path}")
    ########################
    ########################
    @property
    def name(self):
        # define name and path of savefile
        return f"Sub_{self.Gal_name}_Npix{self.pixel_num}_Part{self.PartLens_name}"
    @property
    def pkl_path(self):
        return self.savedir/f"{self.name}.pkl"

    def is_precomputed(self):
        if self.pkl_path.exists():
            return True
        return False
    #############################
    # Run:
    def upload_prev(self):
        if not self.reload:
            return False
        prev_mod = ReadLens(self)
        if prev_mod is False or prev_mod != self:
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_mod.__dict__}
        return True

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
            
    def galaxy_projection(self,verbose=True):            
        # Read particles ONCE
        # kwargs of Msun, XYZ in kpc (explicitely) centered around Centre of Mass (CM)
        kw_parts         = Gal2kwMXYZ(self.Gal) 
        # Compute projection
        kwres_proj_res    = projection_main_AMR(Gal=self.Gal,kw_parts=kw_parts,
                                               z_source_max=self.z_source_max,
                                               sample_z_source=self.sample_z_source,
                                               min_thetaE=self.min_thetaE,
                                               arcXkpc=self.arcXkpc,verbose=verbose,
                                               reload=self.reload)
        # load latest successful projection
        kwres_proj = kwres_proj_res["projs"][-1]
        # store results
        self.proj_index   = kwres_proj["proj_index"]
        self.z_source_min = kwres_proj["z_source_min"]
        self.z_source     = kwres_proj["z_source"]
        self.MD_coords    = kwres_proj["MD_coords"]
        if verbose:
            print("Z source sampled:",self.z_source)
        self.thetaE    = kwres_proj["thetaE"]
        if verbose:
            print("Approx. thetaE:",np.round(self.thetaE,3))
   
        # the following can only be computed once we know the z_source:
        self.SigCrit       = SigCrit(cosmo=self.cosmo,
                                     z_lens=self.z_lens,
                                     z_source=self.z_source) # Msun/kpc^2
    @property
    def deltaPix(self):
        Diam_arcsec = 2*self.radius #diameter in arcsec
        deltaPix    = Diam_arcsec/self.pixel_num # ''/pix
        return deltaPix

    def create_lens(self,verbose=True):
        # setup lensing keywords
        self.PartLens.setup(self) # only run now as it needs z_source 
        # Define the radius based on ~ theta_E
        scale_tE       = 2 
        self.radius    = self.thetaE*scale_tE
        if verbose:
            print("Image radius:",np.round(self.radius,3))
        # setup grid:
        self.grid
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
        return 0
        
    def sample_z_source(self,z_source_min,z_source_max):
        # this is here to allow modularity 
        if self.kw_like_zs is None:
            # simple uniform sample btw the ranges
            z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        elif "fixed" in self.kw_like_zs.keys():
            # if fixed, we don't sample it
            z_source = self.kw_like_zs["fixed"]
            # ensure it is still acceptable
            assert z_source>=z_source_min and z_source<z_source_max
        else:
            # else we follow the given likelihood
            Lkl_source = Likelihood(var_range=[[z_source_min,z_source_max]],
                                    kw_like = self.kw_like_zs)
            # the following still has shape = n_walkers
            z_source_list = Lkl_source.sample(n_samples=1,progress=False)
            z_source      = np.random.choice(z_source_list)
        return z_source
    #
    # Lensing computations
    #
    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)
    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
    @cached_property
    def hessian(self):
        return self._hessian(_radec=None)
    @cached_property
    def psi_map(self):
        return self._psi_map(_radec=None)
        
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
            kw_extents = cod.get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
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
    #
    # Coordinates 
    # 
    @property
    def _radec(self):
        return make_grid(self.pixel_num,to_dimless(self.deltaPix))
    
    """
    @property
    def grid(self):
        transform_pix2angle = np.array([[to_dimless(self.deltaPix),0],
                                        [0,to_dimless(self.deltaPix)]] )
        radec_at_xy_0       = -to_dimless(self.deltaPix)*(self.pixel_num-1)/2
        grid = RegularGrid(
                self.pixel_num,
                self.pixel_num,
                transform_pix2angle,
                radec_at_xy_0,
                radec_at_xy_0,
                supersampling_factor=1,
                flux_evaluate_indexes=None,
            )
        return grid
    """
    @property
    def kw_extents(self):
        kw_extents = cod.get_extents(arcXkpc=self.arcXkpc,Model=self,
                                     _radec=self._radec)
        return kw_extents
        
    def get_RADEC(self):
        _ra,_dec = self._radec
        RA,DEC   = array2image(_ra),array2image(_dec)
        return RA,DEC
