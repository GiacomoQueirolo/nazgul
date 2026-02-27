"""
From given galaxies, read particles and generate Lens Class
Revolves around the SubLensPart class, plus several helper functions

-> restructured to be "subdominant" w.r.t. LensModel
     -> no additional lens profiles possible (at this level)
     -> loose z_lens and z_source(=np.inf) (but store the details for later use)
"""

import dill
import numpy as np
from pathlib import Path
from functools import cached_property 

import astropy.units as u
from lenstronomy.Util import util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.data_api import DataAPI
# My libs
from python_tools.get_res import LoadClass
from python_tools.tools import mkdir,to_dimless,ensure_unit
# cosmol. params.
from nazgul.lib_cosmo import SigCrit
# Get particle from galaxy catalogue
from nazgul.particle_galaxy import get_rnd_PG,Gal2kwMXYZ,LoadGal
# particle lens class and params.
from nazgul.particle_lenses import PMLens 
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
# likelihood class
from nazgul.likelihood import Likelihood
from nazgul.likelihood_z_source import kw_prior_z_source_zl
# project galaxy along various axis
from nazgul.project_gal import get_2Dkappa_map,ProjGal,projection_main_AMR
from nazgul.project_gal import Gal2kw_samples,ProjectionError

# default parameters:
pixel_num     = 200 # pix for image
verbose       = True
# for z_source computation:
z_source_max  = 4
#minimum theta_E
min_thetaE = .3*u.arcsec #arcsec

# Path definitions:

# define where to store the obtained lenses classes
std_sim_lens_path   = Path("./sim_lens/")
default_savedir_sim = "test_sim_lens_AMR" # subdirectory depending on the lensing algorithm

def get_lens_dir(Gal,sim_lens_path=std_sim_lens_path):
    #lens_dir = f"{sim_lens_path}/{Gal.sim}/snap{Gal.snap}_G{Gal.Gn}.{Gal.SGn}/"
    lens_dir = Path(sim_lens_path)/Gal.sim/f"snap{Gal.snap}_G{Gal.Gn}.{Gal.SGn}"
    mkdir(lens_dir)
    Gal.lens_dir = lens_dir
    return lens_dir
##########################
# Model class for parts. #
##########################
# kwargs of ultra-performing band for default simulated images -> quite arbitrary, possibly to improve
kwargs_band_sim = {'read_noise': 0, # no RN noise
 'pixel_scale': None,               # to update depending on the lens
 'ccd_gain': 2.5,             # standard gain for HST
 'exposure_time': 5400.0,     # standard exp time for HST
 'sky_brightness': 35,        #"dark" sky
 'magnitude_zero_point': 30,  # very deep 
 'num_exposures': 1,          # standard HST n exp.
 'psf_type': 'NONE'}          # "infinite" psf resolution 

##########################
# Model class for parts. #
##########################
# kwargs of ultra-performing band for default simulated images -> quite arbitrary, possibly to improve
kw_prior_z_source_minimal = {"z_source_max":z_source_max}
kw_prior_z_source_stnd    = kw_prior_z_source_zl|kw_prior_z_source_minimal

class SubLensPart(): 
    def __init__(self,
                 Galaxy,      # class instance of PartGal
                 kwlens_part, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # number of pixels 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE, # minimum theta observable
                 kwargs_band_sim=kwargs_band_sim, #simulation properties
                 savedir_sim="lensing",           # where to store
                 sim_lens_path=std_sim_lens_path, # where to load the data
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
                raise RuntimeError("Previously defined as not a lens")
        
        lens_dir           = get_lens_dir(self.Gal,sim_lens_path=sim_lens_path)
        self.savedir_sim   = savedir_sim
        self.savedir       = lens_dir/savedir_sim
        self.reload        = reload
        mkdir(self.savedir)
        ######
        # lensing params
        self.pixel_num     = pixel_num      
        self.kwlens_part   = kwlens_part
        self.PMLens        = PMLens(kwlens_part)
        self.PMLens_name   = self.PMLens.name
        # cosmo params
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # criteria for supercriticality of the lens
        self.z_source_max  = z_source_max
        self.kw_like_zs    = kw_prior2like_zs(kw_prior_z_source=kw_prior_z_source,
                                              z_lens=self.z_lens)
        self.min_thetaE    = ensure_unit(min_thetaE,u.arcsec) #arcsec
        
        # observational params
        self.kwargs_band_sim     = kwargs_band_sim
        
        # source model - probably to improve
        self.kwargs_source_model = {"source_light_model_list":source_model_list,
                                   "cosmo":self.cosmo
                                   }


    ### Class Structure ####
    ########################
    def _identity(self):
        """Return an immutable tuple uniquely identifying this galaxy.

        The identity is used for hashing, equality, and cache keys.
        """
        return (
            self.Gal._identity(),
            self.PMLens._identity(),
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
        if not isinstance(other, LensPart):
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
        state.pop('imageModel',None)
        # These are reloaded / reconstructed
        state.pop('Gal',None)
        state.pop('PMLens',None)
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
            for attr in ("Gal", "PMLens", "cosmo")
        )
    def _unpack(self):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        # reload Galaxy
        Galaxy = LoadGal(self.Gal_path)
        Galaxy = ProjGal(Galaxy)
        self.Gal      = Galaxy
        self.Gal_name = Galaxy.Name
        self.cosmo    = Galaxy.cosmo
        
        # re-define PMLens
        self.PMLens = PMLens(self.kwlens_part)
        self.PMLens.setup(self)
        
        # Rebuild lens model if missing
        if not hasattr(self, "lens_prof"):
            self.setup_lenses()

        # Rebuild image model
        if not hasattr(self, "imageModel"):
            self.set_imageModel()
        
        
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
        return f"Sub_{self.Gal_name}_Npix{self.pixel_num}_Part{self.PMLens_name}"
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
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            # Lens Verification:
            ####################
            # project and check if it is a lens
            self.galaxy_projection(verbose=verbose)

            # Lensing computations
            ######################
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
        # Define the radius based on ~ theta_E
        scale_tE       = 2 
        self.radius    = self.thetaE*scale_tE
        if verbose:
            print("Image radius:",np.round(self.radius,3))
   
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
        self.PMLens.setup(self) # only run now as it needs z_source 
        self.setup_dataclasses()
        # setup imageModel:
        self.set_imageModel()
        # setup lenses 
        self.setup_lenses()
        # compute alpha map (most computationally intense function):
        self.alpha_map
        self.kappa_map
        self.hessian
        self.psi_map
        # Store the results
        self.store()

    def setup_lenses(self):
        """
        Setup lensing parameters tailored for lenstronomy
        first compute the particle lenses
        if present, add the additional lenses components
        """
        self.setup_particle_lenses()
        return 0
        
    # the following is meant to be rerun every time we load the class to save space
    # -> computationally not intense 
    def setup_particle_lenses(self):
        print("Setting up lensing parameters...")
        # Convert x,y,z in samples and get masses
        kw_samples = Gal2kw_samples(Gal=self.Gal,proj_index=self.proj_index,
                                    MD_coords=self.MD_coords,arcXkpc=self.arcXkpc)
        samples    = kw_samples["RAs"],kw_samples["DECs"]
        Ms         = kw_samples["Ms"]
        # Convert in lenses parameters 
        kwLnsPart,LnsProfPart   = self.PMLens.get_lens_PART(samples=samples,Ms=Ms)
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

    def setup_dataclasses(self):
        """
        Define all classes not dependent on lensing
        """
        self.kwargs_band_sim["pixel_scale"] = to_dimless(self.deltaPix)
        dataAPI              = DataAPI(self.pixel_num,**self.kwargs_band_sim)    
        self.data_class      = dataAPI.data_class()(**self.kwargs_data)
        self.kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}    
        return 0
        
    def set_imageModel(self):
        self.imageModel = self.get_imageModel()
        return self.imageModel

    def get_imageModel(self):
        # use default kwargs_numerics
        if not hasattr(self, "kwargs_numerics") or not hasattr(self, "data_class"):
            self.setup_dataclasses()
        if not hasattr(self, "lens_model"):
            self.setup_lenses()
        
        imageModel = ImageModel(self.data_class, psf_class=None, 
                            lens_model=self.lens_model, 
                            source_model_class=None,
                            lens_light_model_class=None,
                            point_source_class=None, 
                            kwargs_numerics=self.kwargs_numerics)
        return imageModel

    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)
    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
    @cached_property
    def hessian(self):
        return self._hessian()
    @cached_property
    def psi_map(self):
        psi = self._psi_map(_radec=None)
        return psi
        
    def _psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_prof.function(_ra, _dec, **self.kwargs_lens)
        psi = util.array2image(psi)
        return psi
        
    def _alpha_map(self,_radec=None):
        print("Computing lensing PM deflection...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_prof.derivatives(_ra, _dec, **self.kwargs_lens)
        alpha_x,alpha_y = util.array2image(alpha_x),util.array2image(alpha_y)
        return alpha_x,alpha_y
    """
    def _kappa_map_from_lens(self,_radec=None,exact=False):
        # compute analytically from the particles -> actually should not be the way to do it !
        print("Computing kappa map from PM...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        f_xx,f_xy,f_yx,f_yy= self.lens_prof.hessian(_ra, _dec, **self.kwargs_lens)
        kappa = 1.0 / 2 * (f_xx + f_yy)
        kappa = util.array2image(kappa)
        return kappa
    """ 
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
        

    # Shear components and caustics/CL
    def _hessian(self):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        -> cannot be done outside the grid
        """
         # Note: this hessian only consider the contribution of the alpha map within the cutout!
        alpha_x,alpha_y = self.alpha_map
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, to_dimless(self.deltaPix))
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, to_dimless(self.deltaPix))
        #print("Note: Taking the average of dalpha_x_dy and dalpha_y_dx for fxy")
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy
    #
    # Coordinates 
    # 
    @property
    def kw_extents(self):
        _radec = self.imageModel.ImageNumerics.coordinates_evaluate
        kw_extents = get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        return kw_extents
        
    def get_RADEC(self):
        self.unpack()
        _ra,_dec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        RA,DEC   = util.array2image(_ra),util.array2image(_dec)
        return RA,DEC

 

#
# helper funct
#

def kw_prior2like_zs(kw_prior_z_source,z_lens):
    """
    Convert kw_prior_z_source into the kw_like 
    needed for the Likelihood class for the z_source sampling
    if kw_prior_z_source does not have the required function, returns None
    if kw_prior_z_source had "fixed_z_source", fix the z_source
    """
    kw_like_zs = None
    prior_keys = kw_prior_z_source.keys()
    if "f_lkl_z_source" in prior_keys and "fixed_z_source" in prior_keys:
        raise RuntimeError("Either sample z_source or fix it")
    elif "f_lkl_z_source" in prior_keys:
        kw_like_zs = {}
        # wrapper funct. to fix the z_lens
        def like_func_zs(z_source,*args):
            f_lkl_pr = kw_prior_z_source["f_lkl_z_source"]
            return f_lkl_pr(z_source=z_source,z_lens=z_lens,*args)
        kw_like_zs["like_func"] = like_func_zs
        kw_like_zs["like_prms"] = kw_prior_z_source.get("prms_lkl_z_source",[]) 
        return kw_like_zs
    elif "fixed_z_source" in prior_keys:
        def lkl(zs,*args):
            return 0 
        kw_like_zs = {"fixed":kw_prior_z_source["fixed_z_source"]}
    return kw_like_zs
        

# we override the standard loading function to recompute some large dataset that
# are deleted to save space
def LoadLens(LnsCl,verbose=True):
    LnsCl = LoadClass(LnsCl,verbose=verbose)
    # has to consider the possibility it failed to load
    if LnsCl: 
        # recompute deleted components
        LnsCl.unpack()
    return LnsCl
    
def ReadLens(aClass,verbose=True):
    return LoadLens(aClass.pkl_path,verbose=verbose)


def get_extents(arcXkpc,Model=None,_radec=None):
    """Returns the extent of the image in various units
    """
    if _radec is None:
        _radec = Model.imageModel.ImageNumerics.coordinates_evaluate #arcsecs 
    _ra,_dec = _radec
    RA,DEC   = util.array2image(_ra),util.array2image(_dec)
    ra0,dec0 = RA[0]*u.arcsec,DEC.T[0]*u.arcsec # center of the bin

    Dra0  = np.diff(ra0)  
    Ddec0 = np.diff(dec0)

    ra_edges  = np.hstack([ra0[0]-(Dra0[0]/2.),ra0[1:]-(Dra0/2.),ra0[-1]+.5*Dra0[-1]])
    dec_edges = np.hstack([dec0[0]-(Ddec0[0]/2.),dec0[1:]-(Ddec0/2.),dec0[-1]+.5*Ddec0[-1]])
    
    Dra01   = np.diff(ra_edges) 
    # ugly, could be cleaner-> but after all it's constant so 
    Ddec01  = np.diff(dec_edges)

    # in kpc:
    xmin = ra_edges[0]/arcXkpc
    xmax = ra_edges[-1]/arcXkpc
    ymin = dec_edges[0]/arcXkpc
    ymax = dec_edges[-1]/arcXkpc

    extent_kpc    = [xmin.value, xmax.value,ymin.value, ymax.value] #kpc
    extent_arcsec = [ra_edges[0].value, ra_edges[-1].value,dec_edges[0].value, dec_edges[-1].value] #arcsec
    bins_arcsec   = [ra_edges,dec_edges]
    kw_extents = {"extent_kpc":extent_kpc,
              "extent_arcsec":extent_arcsec,
              "bins_arcsec":bins_arcsec,
              "DRaDec":[Dra01,Ddec01]}
    return kw_extents

# get a lens no matter what:
def wrapper_get_rnd_sublens(reload=True,
                        kw_lenspart={}):
    """Try to get a lens from random galaxies, repeat until finds one
    which is an actual lens (i.e. supercritical)
    """
    
    default_kw_lenspart={"kwlens_part":kwlens_part_AS,
                     "kw_prior_z_source":kw_prior_z_source_minimal,
                     "pixel_num":pixel_num,
                     "reload":reload,
                     "savedir_sim":default_savedir_sim}
    kw_lenspart = default_kw_lenspart.update(kw_lenspart)
    while True:
        Gal    = get_rnd_PG()
        mod_LP = LensPart(Galaxy=Gal,
                          **kw_lenspart)
        try:
            mod_LP.run()
            break
        except ProjectionError as PE:
            print("This galaxy failed: ",PE,"\n","Trying different galaxy")
            pass
    return mod_LP


if __name__ == "__main__":
    print("Do not run this script, but test_generate_particle_lens.py")
    exit()