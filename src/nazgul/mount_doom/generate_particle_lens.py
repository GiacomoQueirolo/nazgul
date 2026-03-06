"""
Older version of the approach, mantained for now. Aim to phase it out in favour of a combination of sub/LensModel/dom

From randomly selected galaxies, read particles and generate Particles Lenses
Revolves around the LensPart class, plus several helper functions -> imported by cracks_of_doom
"""

import dill
import numpy as np
from functools import cached_property 

import astropy.units as u
from scipy.ndimage import zoom
from scipy.interpolate import  RectBivariateSpline

from lenstronomy.Util import util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.sim_api import SimAPI

# My libs
from python_tools.tools import mkdir,to_dimless,ensure_unit
# cosmol. params.
from nazgul.lib_cosmo import SigCrit
# Get particle from galaxy catalogue
from nazgul.particle_galaxy import Gal2kwMXYZ,LoadGal
# particle lens class and params.
from nazgul.particle_lenses import  PartLensExpanded
# likelihood class
from nazgul.likelihood import Likelihood
# project galaxy along various axis
from nazgul.project_gal import get_2Dkappa_map,ProjGal,projection_main_AMR
from nazgul.project_gal import Gal2kw_samples

from nazgul.mount_doom.cracks_of_doom import pixel_num,z_source_max,min_thetaE
from nazgul.mount_doom.cracks_of_doom import std_sim_lens_path,default_savedir_sim
from nazgul.mount_doom.cracks_of_doom import kwargs_source_default,source_model_list
from nazgul.mount_doom.cracks_of_doom import kwargs_band_sim,kw_prior_z_source_stnd,kw_prior_z_source_minimal
from nazgul.mount_doom.cracks_of_doom import ReadLens
import nazgul.mount_doom.cracks_of_doom as cod

verbose = True

class LensPart(): 
    def __init__(self,
                 Galaxy,
                 kwlens_part, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # sim prms 
                 kw_add_lenses=None, # additional lenses (e.g. LOS)
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE,
                 source_model_list=source_model_list, # this might not be the most efficient way to do it..
                 kwargs_band_sim=kwargs_band_sim,
                 savedir_sim="lensing",
                 sim_lens_path=std_sim_lens_path,
                 reload=True # reload previous lens
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
        
        lens_dir           = cod.get_lens_dir(self.Gal,sim_lens_path=sim_lens_path)
        self.savedir_sim   = savedir_sim
        self.savedir       = lens_dir/savedir_sim
        self.reload        = reload
        mkdir(self.savedir)
        ######
        # lensing params
        self.pixel_num     = pixel_num      
        self.kwlens_part   = kwlens_part
        self.kw_add_lenses = kw_add_lenses
        self.PartLens        = PartLensExpanded(kwlens_part,kw_add_lenses=kw_add_lenses)
        self.PartLens_name   = self.PartLens.name
        # cosmo params
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # criteria for supercriticality of the lens
        self.z_source_max  = z_source_max
        self.kw_like_zs    = cod.kw_prior2like_zs(kw_prior_z_source=kw_prior_z_source,
                                              z_lens=self.z_lens)
        self.min_thetaE    = ensure_unit(min_thetaE,u.arcsec) #arcsec
        
        # observational params
        self.kwargs_band_sim     = kwargs_band_sim
        # source model - probably to improve
        self.kwargs_source_model = {"source_light_model_list":source_model_list,
                                   "cosmo":self.cosmo
                                   }
        #self.Sim = None  will be initialised with kwargs_band_sim once we have the correct deltaPix

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
            self.kw_add_lenses
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
        state.pop('lens_model', None)
        state.pop('kw_shear',None)
        state.pop('imageModel',None)
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
        # reload Galaxy
        Galaxy = LoadGal(self.Gal_path)
        Galaxy = ProjGal(Galaxy)
        self.Gal   = Galaxy
        self.Gal_name = Galaxy.Name # unsure why this is needed
        self.cosmo = Galaxy.cosmo
        
        # re-define PartLens
        self.PartLens = PartLensExpanded(self.kwlens_part,kw_add_lenses=self.kw_add_lenses)
        self.PartLens.setup(self)
        
        # Rebuild lens model if missing
        if not hasattr(self, "lens_model"):
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
        return f"{self.Gal_name}_Npix{self.pixel_num}_Part{self.PartLens_name}"
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
        # update kwargs_band_sim:
        self.kwargs_band_sim["pixel_scale"] = to_dimless(self.deltaPix)
        # define Sim for the lens computations
        # self.Sim DOES NOT contain lensing information, it's used only partially and carefully
        self.Sim = SimAPI(numpix=self.pixel_num,
                          kwargs_single_band=self.kwargs_band_sim,
                          kwargs_model=self.kwargs_source_model,
                          )
        # in a similar way a posterior SimObs can be used to create images given different telescopes
        
        # setup dataclasses (dataclass,psf_class,sourcemodel and some helper kwargs):
        self.setup_dataclasses()
        # setup imageModel:
        self.set_imageModel()
        # setup lenses 
        self.setup_lenses()
        # compute alpha map (most computationally intense function):
        self.alpha_map
        self.sample_source_pos(update=True)
        self.image_sim  = self.get_lensed_image()
        # Store the results
        self.store()

    def setup_dataclasses(self,Sim=None):
        """
        Define all classes not dependent on lensing
        Handled by SimAPI
        """
        if Sim is None:
            Sim = self.Sim
        self.data_class,self.psf_class,self.source_model_class,self.kwargs_numerics,self.kwargs_source = cod.get_dataclasses(Sim)
        return 0
        
    def update_source_position(self,ra_source,dec_source):
        # useful if we want to put it in the center of the caustic
        self.kwargs_source["center_x"]= ra_source
        self.kwargs_source["center_y"]= dec_source
        return 0

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
        kwLnsPart,LnsModPart   = self.PartLens.get_lens_PART(samples=samples,Ms=Ms)
        self.kwargs_lens       = kwLnsPart
        self.lens_model        = LnsModPart
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
        
    def get_lensed_image(self,imageModel=None,
                         sourceModel=None,kwargs_source=None,\
                         alpha_map=None, # can input directly the alpha map
                         unconvolved=True):
        self.unpack()
        if sourceModel is None:
            sourceModel  = self.source_model_class
        if kwargs_source is None:
            kwargs_source = self.kwargs_source                
        if imageModel is None:
            imageModel   = self.imageModel
        
        x_source_plane,y_source_plane = self.get_xy_source_plane(alpha_map=alpha_map)
        kwargs_source_list = [kwargs_source]
        source_light = sourceModel.surface_brightness(x_source_plane, y_source_plane, kwargs_source_list, k=None)
        if not np.abs(imageModel.Data.pixel_width-to_dimless(self.deltaPix))<1e-10:
            if imageModel.Data.pixel_width<to_dimless(self.deltaPix):
                raise RuntimeError("Simulated observatory cannot have higher resolution than baseline simulation") 
            # different pixel scale (due different simulated observatory)
            source_light_im = util.array2image(source_light)
            pixel_num_aim = imageModel.ImageNumerics.grid_class.num_grid_points_axes[0]
            source_light_downgrade = zoom(source_light_im,pixel_num_aim/self.pixel_num)
            source_light = util.image2array(source_light_downgrade)
        image_sim    = imageModel.ImageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return image_sim


    def sample_source_pos(self,update=False):
        """Sample the source position within the tangential
        critical caustic
        """
        kw_caustics  = self.critical_curve_caustics
        ra_ct,dec_ct = kw_caustics["caustics"]["tangential"]
        print("Sampling source position within tangential caustic")
        # the tangential caustic is approximated to circular
        # and we sample uniformily within that 
        ra0_ct,dec0_ct = np.mean(ra_ct),np.mean(dec_ct)
        rads_ct = np.hypot(ra_ct-ra0_ct,dec_ct-dec0_ct)
        rad0_ct = np.std(rads_ct)
        rad_source = np.random.uniform(0,rad0_ct)
        phi_source = np.random.uniform(0,2*np.pi)
        ra_source  = rad_source*np.cos(phi_source) 
        dec_source = rad_source*np.sin(phi_source) 
        if update:
            self.update_source_position(ra_source,dec_source)
        return ra_source,dec_source

    def get_xy_source_plane(self,alpha_map=None):
        """Map the x,y grid into the source plane
        (used to fit the light of the source to the image)
        """
        RA,DEC       = self.get_RADEC()
        # if not given as input, reads the computed deflection map
        if alpha_map is None:
            # if not already, compute alpha_map
            alpha_map = self.alpha_map
        alpha_x,alpha_y = alpha_map        
        x_source_plane, y_source_plane = RA-alpha_x,DEC-alpha_y
        # the coords have to be given as flat
        x_source_plane = util.image2array(x_source_plane)
        y_source_plane = util.image2array(y_source_plane)
        return x_source_plane,y_source_plane
        
    def get_imageModel(self,Sim=None):
        # use default kwargs_numerics
        if not hasattr(self, "kwargs_numerics"):
            self.setup_dataclasses(Sim=Sim)
        if not hasattr(self, "lens_model"):
            self.setup_lenses()
            
        if Sim is None or Sim==self.Sim:
            Sim = self.Sim
            if not hasattr(self, "data_class"):
                self.setup_dataclasses(Sim=Sim)
            data_class = self.data_class
            psf_class  = self.psf_class
            source_model_class = self.source_model_class
        else:
            data_class,psf_class,source_model_class,_,_ = cod.get_dataclasses(Sim)
        
        imageModel = ImageModel(data_class, psf_class, 
                            self.lens_model, 
                            source_model_class,
                            lens_light_model_class=None,point_source_class=None, 
                            kwargs_numerics=self.kwargs_numerics)
        return imageModel
        
    def set_imageModel(self):
        self.imageModel = self.get_imageModel(self.Sim)
        return self.imageModel
    
    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)

    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
    @cached_property
    def hessian(self):
        return self._hessian()
    
    def compute_psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        self.unpack()
        if _radec is None:
            # equivalent to np.reshape(np.array(self.data_class.pixel_coordinates),(2,self.pixel_num*self.pixel_num))
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_model.potential(_ra, _dec, self.kwargs_lens)
        psi = util.array2image(psi)
        return psi
        
    def _alpha_map(self,_radec=None):
        print("Computing lensing PM deflection...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_model.alpha(_ra, _dec, self.kwargs_lens)
        alpha_x,alpha_y = util.array2image(alpha_x),util.array2image(alpha_y)
        return alpha_x,alpha_y
        
    def _kappa_map_from_lens(self,_radec=None,exact=False):
        # compute analytically from the particles -> actually should not be the way to do it !
        print("Computing kappa map from PM...")
        self.unpack()
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        kappa = self.lens_model.kappa(_ra, _dec, self.kwargs_lens)
        kappa = util.array2image(kappa)
        return kappa
        
    def _kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        print("Computing kappa map from density map...")
        if _radec is None:
            kw_extents = self.kw_extents
        else:
            kw_extents = cod.get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        kappa = get_2Dkappa_map(Gal=self.Gal,proj_index=self.proj_index,MD_coords=self.MD_coords,kwargs_extents=kw_extents,
                                SigCrit=self.SigCrit,arcXkpc=self.arcXkpc)
        return kappa
        
    @cached_property
    def psi_map(self):
        psi = self.compute_psi_map(_radec=None)
        self.store()
        return psi        
    #
    # Coordinates 
    # 
    @property
    def kw_extents(self):
        _radec = self.imageModel.ImageNumerics.coordinates_evaluate
        kw_extents = cod.get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        return kw_extents
        
    def get_RADEC(self):
        self.unpack()
        _ra,_dec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        RA,DEC   = util.array2image(_ra),util.array2image(_dec)
        return RA,DEC

    ################
    ################
    # Simulating observations: 

    # band will have to have also psf information!
    def get_SimObs(self,band,
                   kwargs_psf=None,# add psf and pssf
                   kwargs_source_model=None):
        if kwargs_source_model is None:
            kwargs_source_model = self.kwargs_source_model
        kwargs_model = {"z_source":self.z_source} | kwargs_source_model
        if not "cosmo" in kwargs_model.keys():
            # cosmology should in principle not be used, but better be consistent
            kwargs_model["cosmo"] = self.cosmo
        #self.bandObs = band # so we know which one it is running
        
        # realistic observation
        kwargs_single_band = band.kwargs_single_band()
        if kwargs_psf is not None:
            if not kwargs_psf.keys() == {"kernel_point_source":[],"point_source_supersampling_factor":[]}.keys():
                raise RuntimeError(f"kwargs_psf has to have only kernel_point_source and point_source_supersampling_factor, not {kwargs_psf.keys()}")
            kwargs_single_band.update(kwargs_psf)
        # must recompute pixel_num in order to covert to ~ the same aperture,
        # but with the new resolution 
        # -> round down to be sure we are within the bounds
        pixel_num = int(to_dimless(2*self.radius)/kwargs_single_band["pixel_scale"])
        
        # instantiate simulation API class
        SimObs = SimAPI(numpix = pixel_num, # N of pixels in "observed" image
                     kwargs_single_band = kwargs_single_band, # telescope specific keyword arguments (eg HST, see above)
                     kwargs_model = kwargs_model,# kwargs source model (in principle kw lens as well)
                    )
        return SimObs

    def sim_image(self,SimObs,noisy=False):
        """Obtain simulated images given SimObs
        """
        imageModelObs    = self.get_imageModel(Sim=SimObs)
        sourceModelObs   = SimObs.source_model_class
        kwargs_sourceObs = cod.get_kwargs_sourceSim(SimObs)    

        image_SimObs     = self.get_lensed_image(imageModel=imageModelObs,
                                             sourceModel=sourceModelObs,
                                             kwargs_source=kwargs_sourceObs,
                                             unconvolved=False)
        if noisy:
            image_SimObsnoisy  = image_SimObs + SimObs.noise_for_model(model=image_SimObs)
            error_SimObs       = SimObs.estimate_noise(image_SimObsnoisy) # NOT variance
            return image_SimObsnoisy,error_SimObs
        return image_SimObs

    def sim_multi_band_list(self,band,kwargs_psf=None,kwargs_source_model=None):
        """Setup Simulation given band specific, its psf and kwargs_source_model 
        """
        SimObs                         = self.get_SimObs(band,kwargs_psf=kwargs_psf,
                                                         kwargs_source_model=kwargs_source_model)
        image_SimObsnoisy,error_SimObs = self.sim_image(SimObs,noisy=True)
        kw_data_sim = SimObs.kwargs_data
        kw_data_sim["image_data"] = image_SimObsnoisy
        kw_data_sim["noise_map"] = error_SimObs
        kw_psf_sim = SimObs.kwargs_psf
        # consider if to add psf error - depends on observations
        if "point_source_supersampling_factor" in kw_psf_sim.keys():
            kw_numerics_sim = {'point_source_supersampling_factor':kw_psf_sim["point_source_supersampling_factor"]} 
        else:
            kw_numerics_sim = {'point_source_supersampling_factor':1}
        image_band      = [kw_data_sim, kw_psf_sim, kw_numerics_sim]
        multi_band_list = [image_band]
        return multi_band_list
        
    # maybe this should be a daughter class
    def get_kw_model_input(self,band,kwargs_source_model=None):
        image,noise_map = self.sim(band=band,noisy=True)

        kw_data = {"kernel_point_source":psf_supersampled,
               "point_source_supersampling_factor":supersampling_factor,
               "psf_error_map":err_psf, #non-supersampled
               "image_data":sim_img_conv_noised,
               "noise_map":noise_map,
               "exp_time":exp_time,
               "bckg_rms":bckg_rms,
               "deltaPix":HST_deltapix}
    ################
    ################

    # Shear components and caustics/CL
    def _hessian(self):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        """
         # Note: this hessian only consider the contribution of the alpha map within the cutout!
        alpha_x,alpha_y = self.alpha_map
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, to_dimless(self.deltaPix))
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, to_dimless(self.deltaPix))
        #print("Note: Taking the average of dalpha_x_dy and dalpha_y_dx for fxy")
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy

    def get_kw_shear(self):
        """From the hessian matrix compute the shear components
        """
        f_xx,f_xy,f_yx,f_yy = self.hessian
        # derived shear1,shear2 and shear
        shear1 = 1./2 * (f_xx - f_yy)
        shear2 = f_xy
        shear  = np.hypot(shear1,shear2)
        self.kw_shear = {"shear1":shear1,"shear2":shear2,"shear":shear}
        return self.kw_shear
        
    @property
    def shear_map(self):
        if hasattr(self,"kw_shear"):
            return self.kw_shear["shear"]
        return self.get_kw_shear()["shear"]

    @cached_property
    def critical_curve_caustics(self):
        return self.get_kw_critical_curve_caustics()

    def get_kw_critical_curve_caustics(self):
        """ Fit the critical curve and map it to the caustic
        """
        # note: alpha is computed from the particle (and shear from alpha)
        # thus depends on the particle lens model chosen, while kappa
        # is obtained directly as a density map + cosmological scaling
        alpha_x,alpha_y = self.alpha_map
        kappa           = self.kappa_map
        shear           = self.shear_map
        
        eigen_rad = 1 - kappa + shear
        eigen_tan = 1 - kappa - shear
        # have to find when those are ~0
        mintv = np.min(np.abs(eigen_rad))
        Dv    = np.max(np.abs(eigen_rad)) - mintv
        maxtv = mintv + 0.1*Dv
        test_values = np.linspace(mintv,maxtv,20)
    
        ierx,ietx = [],[] #placeholders 
        # radial
        for tv in test_values:
            if len(ierx)/(self.pixel_num**2) >0.001 :
                break
            else:
                iery,ierx = np.where(cod.MAD_mask(np.abs(eigen_rad),0,tv)) 
        # tangential
        mintv = np.min(np.abs(eigen_tan))
        Dv = np.max(np.abs(eigen_tan)) - mintv
        maxtv = mintv + 0.1*Dv
        test_values = np.linspace(mintv,maxtv,20)
        for tv in test_values:
            if len(ietx)/(self.pixel_num**2) >0.001:
                break
            else:
                iety,ietx = np.where(cod.MAD_mask(np.abs(eigen_tan),0,tv))
        # coords
        RA,DEC      = self.get_RADEC()
        ra0,dec0    = RA[0],DEC.T[0]
        # critical and caustics divided in tangential and radial
        cl_rad_x_noisy,cl_rad_y_noisy   = ra0[ierx],dec0[iery]    
        cl_tan_x_noisy,cl_tan_y_noisy   = ra0[ietx],dec0[iety]
        # fit them with splines  
        cl_rad_x,cl_rad_y = cod.fit_xy_spline(cl_rad_x_noisy,cl_rad_y_noisy)
        cl_tan_x,cl_tan_y = cod.fit_xy_spline(cl_tan_x_noisy,cl_tan_y_noisy)
        
        # fit alpha in 2D
        # TODO: verify that it is indeed dec0,ra0 and not the other way out ->correct, see TEST
        alpha_x_spline = RectBivariateSpline(dec0,ra0, alpha_x)
        alpha_y_spline = RectBivariateSpline(dec0,ra0, alpha_y)

        cc_rad_x,cc_rad_y   = cl_rad_x-alpha_x_spline.ev(cl_rad_y,cl_rad_x),\
                              cl_rad_y-alpha_y_spline.ev(cl_rad_y,cl_rad_x)

        cc_tan_x,cc_tan_y   = cl_tan_x-alpha_x_spline.ev(cl_tan_y,cl_tan_x),\
                              cl_tan_y-alpha_y_spline.ev(cl_tan_y,cl_tan_x)

        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],
                               "tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],
                                     "tangential":[cl_tan_x,cl_tan_y]}
                  }
        
        return kw_crit

if __name__ == "__main__":
    print("Do not run this script, but test_generate_particle_lens.py")
    exit()