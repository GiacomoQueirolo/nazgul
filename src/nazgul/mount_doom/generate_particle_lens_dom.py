"""
From randomly selected galaxies, read particles and generate Particles Lenses
Revolves around the LensPart class, plus several helper functions

-> restructured to be "dominant" w.r.t. LensModel
    -> set z_lens and sample z_s
    -> can have additional lens models
"""

import dill
import warnings
import numpy as np
from pathlib import Path
from functools import cached_property 
# for debug plots
import matplotlib.pyplot as plt

import astropy.units as u
from scipy.ndimage import zoom
from scipy.interpolate import RectBivariateSpline

from lenstronomy.Util.util import array2image,image2array 
#from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
# My libs
from python_tools.tools import mkdir,to_dimless,convert_error_to_warning
# particle lens class and params.
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
# project galaxy along various axis
from nazgul.project_gal import get_2Dkappa_map,ProjectionError
from nazgul.Translator.translator import get_rnd_PG,get_all_PG
from nazgul.Translator import std_simsuite

from nazgul.mount_doom.generate_particle_lens_sub import SubLensPart
from nazgul.mount_doom.cracks_of_doom import BasicLensPart
from nazgul.mount_doom.cracks_of_doom import pixel_num,min_thetaE
from nazgul.mount_doom.cracks_of_doom import source_model_list
from nazgul.mount_doom.cracks_of_doom import kwargs_band_sim,kw_prior_z_source_stnd
from nazgul.pathfinder import get_lens_highdir_from_galdir

import nazgul.mount_doom.cracks_of_doom as cod

verbose = True
empty_kwargs_add_lenses = {"lens_model_list":[],"kwargs_lens":[]}

class LensPart(BasicLensPart): 
    _large_attributes = ['lens_model','kw_shear', 'lenspart','_Sim','PartLens',
                         'Gal','data_class','psf_class','source_model_class','cosmo']
    def __init__(self,
                 Galaxy,
                 projection_index, # projection index of the galaxy
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # sim prms 
                 kwargs_add_lenses = empty_kwargs_add_lenses, # additional lenses (e.g. LOS)
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE,
                 source_model_list=source_model_list, # this might not be the most efficient way to do it..
                 kwargs_band_sim=kwargs_band_sim,
                 kwargs_lensmodel={}, #additional kwargs to pass to LensModel (e.g. z_lens, as != then of the galaxy one)
                 #subdir="./",
                 reload=True # reload previous lens
                 ):
        """
        :param
            Galaxy :           PartGal instance
            kwlens_part:       kw_args of the PartLens class
            pixel_num:         int, number of pixel
            kwargs_add_lenses: kw_args of 2 elements: lens_model_list (list of name of lens models), 
                                                    kwargs_lens (list of kwargs of lens model parameters)
            kw_prior_z_source: kwargs, define the prior distribution of the z_source to be sampled from
            min_thetaE:        float (ideally with units of arcsec), define the minumum thetaE for which the galaxy is considered a lens
            
        """
        kw_sublenspart = {"Galaxy":Galaxy,"projection_index":projection_index,"kwlens_part":kwlens_part,
                          "pixel_num":pixel_num, "kw_prior_z_source": kw_prior_z_source, 
                          "min_thetaE": min_thetaE,"reload":reload}#"subdir":subdir,
        super().__init__(**kw_sublenspart)
        self.savedir  = get_lens_highdir_from_galdir(galdir=self.Gal.gal_dir)
        mkdir(self.savedir)

        # initialise it independently
        self.lenspart    = SubLensPart(**kw_sublenspart)
        self.id_lenspart = self.lenspart._identity()
        ######
        # lensing params 
        if kwargs_add_lenses is None:
            kwargs_add_lenses    = empty_kwargs_add_lenses
        self.kwargs_add_lenses   = kwargs_add_lenses
        self.kwargs_lensmodel    = kwargs_lensmodel
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
        lnp_id  = self.id_lenspart
        self_id = (self.kwargs_add_lenses,)
        return lnp_id + self_id
 
    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------    
    @classmethod
    def from_SubLens(cls, SubLens):
        """Construct from an existing SimPartGal instance."""
        kw_sublenspart = SubLens.get_kw_sublenspart()
        obj = cls.__init__(**kw_sublenspart)
        return obj
        
    def run(self,read_prev=True,update_source_pos=False,verbose=True):
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            # verify we have the Galaxy
            if verbose:
                print("DEBUG - Unpacking gal (but not load part)...")
            self._unpack_Gal() # now not fully deployed
            if verbose:
                print("DEBUG - Gal unpacked")
            # Lens Verification:
            ####################
            # project and check if it is a lens
            if verbose:
                print("DEBUG - Gal Projection ...")
            self.galaxy_projection(verbose=verbose)
            if verbose:
                print("DEBUG - Projected Gal")
            
            # Lensing computations:
            #######################
            if verbose:
                print("DEBUG - Create lensed...")
            self.create_lens(verbose=verbose)
            if verbose:
                print("DEBUG - Lens created")
            # Image creation:
            #################
            if verbose:
                print("DEBUG - Create Image ...")
            self.sample_source_pos(update=update_source_pos)
            self.image_sim  = self.get_lensed_image()
            if verbose:
                print("DEBUG - Image created")
            self.store()
        
    ########################
    ########################
    @property
    def name(self):
        # define name and path of savefile
        name= f"{self.Gal_name}_Npix{self.pixel_num}_Part{self.PartLens_name}_Prj{self.proj_index}"
        return name
    #############################

    def create_lens(self,verbose=True):
        # Define the radius based on ~ theta_E
        scale_tE       = 2 
        self.radius    = self.thetaE*scale_tE
        if verbose:
            print("Image radius:",np.round(self.radius,3))        
        # setup dataclasses (dataclass,psf_class,sourcemodel and some helper kwargs):
        self.setup_dataclasses()
        # setup lenses 
        self.setup_lenses()
        self.alpha_map

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
        self.kwargs_source["center_x"] = ra_source
        self.kwargs_source["center_y"] = dec_source
        return 0

    def _reformat_kwargs_lensmodel(self):
        # ensure that kwargs_lensmodel has at least 
        # the default z_lens and z_source (if not input differently)
        default_kw_lm = {"z_lens":self.z_lens,
                         "z_source":self.z_source}
        
        kwargs_lensmodel = self.kwargs_lensmodel
        if kwargs_lensmodel is None or kwargs_lensmodel=={}:
            # for consistency, we set z_lens==z_gal and z_source == z_source_sampled
            kwargs_lensmodel     = default_kw_lm
        
        # update default_kw_lm with input kwargs_lensmodel if present
        default_kw_lm |= kwargs_lensmodel
        self.kwargs_lensmodel = default_kw_lm
        
    def setup_lenses(self):
        """
        Setup lensing parameters tailored for lenstronomy
        """
        if hasattr(self,"lens_model") and hasattr(self,"kwargs_lens"):
            print("Lens model already setup")
            return 0
        print("Setting up lensing parameters...")
        add_lens_model_list    = self.kwargs_add_lenses["lens_model_list"]
        add_kwargs_lens        = self.kwargs_add_lenses["kwargs_lens"]
        lens_model_list        = ["PART_GAL",*add_lens_model_list]
        self.kwargs_lens       = [{},*add_kwargs_lens]
        pkwl_part_lens         = {"lenspart":self.lenspart}
        self._reformat_kwargs_lensmodel()
        # the following in principle it is not necessary anymore...?
        if getattr(self.kwargs_lensmodel,"z_lens",self.z_lens)!=self.z_lens:
            pkwl_part_lens["z_lens"] = self.kwargs_lensmodel["z_lens"]
        if getattr(self.kwargs_lensmodel,"z_source",self.z_source)!=self.z_source:
            pkwl_part_lens["z_source"] = self.kwargs_lensmodel["z_source"]
        profile_kwargs_list    = [pkwl_part_lens]
        for adlml in add_lens_model_list:
            profile_kwargs_list.append({})
        #print("profile_kwargs_list",profile_kwargs_list)
        #print("DEBUG",self.kwargs_lensmodel)
        self.lens_model = LensModel(lens_model_list=lens_model_list,
                                    profile_kwargs_list = profile_kwargs_list,
                                    **self.kwargs_lensmodel)
        print("... Lensing parameters set up ")
        
    def get_lensed_image(self,imageNumerics=None,
                         sourceModel=None,kwargs_source=None,
                         alpha_map=None, # can input directly the alpha map
                         unconvolved=True):
        self.unpack()
        self.setup_dataclasses()
        if sourceModel is None:
            sourceModel  = self.source_model_class
        if kwargs_source is None:
            kwargs_source = self.kwargs_source                
        if imageNumerics is None:
            imageNumerics = self.get_imageNumerics(self.Sim)        
        x_source_plane,y_source_plane = self.get_xy_source_plane(alpha_map=alpha_map)
        kwargs_source_list            = [kwargs_source]
        source_light                  = sourceModel.surface_brightness(x_source_plane, y_source_plane, kwargs_source_list, k=None)
        if not np.abs(imageNumerics._pixel_grid.pixel_width-to_dimless(self.deltaPix))<1e-10:
            if imageNumerics._pixel_grid.pixel_width<to_dimless(self.deltaPix):
                raise RuntimeError("Simulated observatory cannot have higher resolution than baseline simulation") 
            # different pixel scale (due different simulated observatory)
            source_light_im = array2image(source_light)
            pixel_num_aim = imageNumerics.grid_class.num_grid_points_axes[0]
            source_light_downgrade = zoom(source_light_im,pixel_num_aim/self.pixel_num)
            source_light = image2array(source_light_downgrade)
        image_sim    = imageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
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
        rads_ct        = np.hypot(ra_ct-ra0_ct,dec_ct-dec0_ct)
        rad0_ct        = np.std(rads_ct)
        rad_source     = np.random.uniform(0,rad0_ct)
        phi_source     = np.random.uniform(0,2*np.pi)
        ra_source      = rad_source*np.cos(phi_source) 
        dec_source     = rad_source*np.sin(phi_source) 
        if self.kwargs_source["center_x"]==0 and self.kwargs_source["center_x"]==0 and not update:
            print("Source position has to be sampled a first time")
            update=True
        if update:
            self.update_source_position(ra_source,dec_source)
        else:
            print("Source position not updated")
        return ra_source,dec_source

    def get_xy_source_plane(self,alpha_map=None):
        """Map the x,y grid into the source plane
        (used to fit the light of the source to the image)
        """
        RA,DEC       = self.get_RADEC()
        # if not given as input, reads the computed deflection map
        if alpha_map is None:
            # if not already, compute alpha_map
            alpha_map   = self.alpha_map
        alpha_x,alpha_y = alpha_map        
        x_source_plane, y_source_plane = RA-alpha_x,DEC-alpha_y
        # the coords have to be given as flat
        x_source_plane = image2array(x_source_plane)
        y_source_plane = image2array(y_source_plane)
        return x_source_plane,y_source_plane
    """
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
    """
    def get_imageNumerics(self,Sim=None):
        if not hasattr(self, "kwargs_numerics"):
            self.setup_dataclasses(Sim=Sim)            
        if Sim is None or Sim==self.Sim:
            Sim = self.Sim
            if not hasattr(self, "data_class"):
                self.setup_dataclasses(Sim=Sim)
            data_class = self.data_class
            psf_class  = self.psf_class
            source_model_class = self.source_model_class
        else:
            data_class,psf_class,source_model_class,_,_ = cod.get_dataclasses(Sim)
        
        imageNumerics = NumericsSubFrame(pixel_grid=data_class,
                                         psf=psf_class, 
                            **self.kwargs_numerics)
        return imageNumerics
        
    def _psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        self.unpack()
        if _radec is None:
            _radec = self._radec #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_model.potential(_ra, _dec, self.kwargs_lens)
        psi = array2image(psi)
        return psi
        
    def _alpha_map(self,_radec=None):
        print("Computing lensing PM deflection...")
        self.unpack()
        if _radec is None:
            _radec = self._radec
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_model.alpha(_ra, _dec, self.kwargs_lens)
        alpha_x,alpha_y = array2image(alpha_x),array2image(alpha_y)
        return alpha_x,alpha_y
        
    def _kappa_map_from_lens(self,_radec=None,exact=False):
        # compute analytically from the particles -> actually should not be the way to do it !
        print("Computing kappa map from PM...")
        self.unpack()
        if _radec is None:
            _radec = self._radec
        _ra,_dec = _radec
        kappa = self.lens_model.kappa(_ra, _dec, self.kwargs_lens)
        kappa = array2image(kappa)
        return kappa
        
    def _kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        print("Computing kappa map from density map...")
        self.unpack()
        if _radec is None:
            kw_extents = self.kw_extents
        else:
            kw_extents = cod.get_extents(arcXkpc=self.arcXkpc,Model=self,_radec=_radec)
        self.Gal.run()
        kappa = get_2Dkappa_map(Gal=self.Gal,proj_index=self.proj_index,MD_coords=self.MD_coords,kwargs_extents=kw_extents,
                                SigCrit=self.SigCrit,arcXkpc=self.arcXkpc)

        if self.kwargs_add_lenses["lens_model_list"]!=[]:
            # Add kappa of the additional profiles
            if _radec is None:
                _radec = self._radec
            _ra,_dec   = _radec
            lens_model_only_add = LensModel(lens_model_list=self.kwargs_add_lenses["lens_model_list"],
                                           **self.kwargs_lensmodel)
            kappa_add  = lens_model_only_add.kappa(_ra, _dec, self.kwargs_add_lenses["kwargs_lens"])
            kappa     += array2image(kappa_add)
        return kappa
        
    ################
    ################
    # Simulating observations: 

    @property 
    def Sim(self):
        if not hasattr(self,"_Sim"):
            self._Sim = self.get_Sim()
        return self._Sim
    
    def get_Sim(self,band=None,
                    kwargs_psf=None,# add psf and pssf
                    kwargs_source_model=None):
        if kwargs_source_model is None:
            kwargs_source_model = self.kwargs_source_model
        kwargs_model = {"z_source":self.z_source} | kwargs_source_model
        if not "cosmo" in kwargs_model.keys():
            # cosmology should in principle not be used, but better be consistent
            kwargs_model["cosmo"] = self.cosmo
        if band is None:
            # This is the resolution of the simulation itself
            kwargs_single_band = self.kwargs_band_sim
            kwargs_single_band["pixel_scale"] = to_dimless(self.deltaPix)
            pixel_num = self.pixel_num
        else:
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
        Sim = SimAPI(numpix = pixel_num, # N of pixels in "observed" image
                 kwargs_single_band = kwargs_single_band, # telescope specific keyword arguments (eg HST, see above)
                 kwargs_model = kwargs_model,# kwargs source model (in principle kw lens as well)
                )
        return Sim

    def sim_image(self,SimObs,noisy=False):
        """Obtain simulated images given SimObs
        """
        imageNumericsObs = self.get_imageNumerics(Sim=SimObs)
        sourceModelObs   = SimObs.source_model_class
        kwargs_sourceObs = cod.get_kwargs_sourceSim(SimObs)    

        image_SimObs     = self.get_lensed_image(imageNumerics=imageNumericsObs,
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
        SimObs                         = self.get_Sim(band=band,kwargs_psf=kwargs_psf,
                                                         kwargs_source_model=kwargs_source_model)
        image_SimObsnoisy,error_SimObs = self.sim_image(SimObs,noisy=True)
        kw_data_sim = SimObs.kwargs_data
        kw_data_sim["image_data"] = image_SimObsnoisy
        kw_data_sim["noise_map"]  = error_SimObs
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
        raise RuntimeError("WIP - prob. to discard")
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
        if len(iery)==0:
            plt.close()
            dbg_plot = "tmp/eigen_rad.png"
            plt.imshow(np.abs(eigen_rad))
            plt.savefig(dbg_plot)
            plt.colorbar()
            plt.close()
            raise RuntimeError(f"Radial critical curve not found.\nCheck {dbg_plot}")
        if len(iety)==0:
            plt.close()
            dbg_plot = "tmp/eigen_tan.png"
            plt.imshow(np.abs(eigen_tan))
            plt.savefig(dbg_plot)
            plt.colorbar()
            plt.close()
            raise RuntimeError(f"Tangential critical curve not found.\nCheck {dbg_plot}")
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

        """
        #TEST: Passed
        i_dec = np.random.choice(np.arange(0,len(dec0)-1))
        i_ra  = np.random.choice(np.arange(0,len(ra0)-1))
        # Direct grid value
        v_grid = alpha_x[i_dec, i_ra]
        
        # Interpolated value at exact grid point
        v_spline = alpha_x_spline.ev(dec0[i_dec], ra0[i_ra])
        
        print(v_grid, v_spline)
        """
        cc_rad_x,cc_rad_y   = cl_rad_x-alpha_x_spline.ev(cl_rad_y,cl_rad_x),\
                              cl_rad_y-alpha_y_spline.ev(cl_rad_y,cl_rad_x)

        cc_tan_x,cc_tan_y   = cl_tan_x-alpha_x_spline.ev(cl_tan_y,cl_tan_x),\
                              cl_tan_y-alpha_y_spline.ev(cl_tan_y,cl_tan_x)

        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],
                               "tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],
                                     "tangential":[cl_tan_x,cl_tan_y]}
                  }
        """
        # DEBUG
        fig,ax = plt.subplots(figsize=(8,8))
        ax.scatter(cc_rad_x,cc_rad_y,c="b",marker=".",label="Radial Caustics")
        ax.scatter(cc_tan_x,cc_tan_y,c="r",marker=".",label="Tangential Caustics")
        ax.scatter(cl_rad_x,cl_rad_y,c="cyan",marker=".",label="Radial Crit. Curve")
        ax.scatter(cl_tan_x,cl_tan_y,c="darkorange",marker=".",label="Tangential Crit. Curve")
        
        #ax.scatter(cc_rad_x_noisy,cc_rad_y_noisy,c="gold",marker=".",label="Radial Caustics noisy")
        #ax.scatter(cc_tan_x_noisy,cc_tan_y_noisy,c="purple",marker=".",label="Tangential Caustics noisy")
        ax.scatter(cl_rad_x_noisy,cl_rad_y_noisy,c="lime",marker=".",label="Radial Crit. Curve noisy")
        ax.scatter(cl_tan_x_noisy,cl_tan_y_noisy,c="peru",marker=".",label="Tangential Crit. Curve noisy")
        
        
        
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        plt.gca().set_aspect('equal')
        ax.set_xlabel("RA ['']")
        ax.set_ylabel("DEC ['']")
        ax.legend()
        ax.set_title("Caustics and Critical Curves") 
        
        plt.tight_layout()
        nm = "tmp/del2.png"
        print(f"Saving {nm}")
        plt.savefig(nm)
        """
        return kw_crit



# get a lens no matter what:
def wrapper_get_rnd_lens(reload=True,
                        kw_lenspart={},
                        kw_galpart={}):
    """Try to get a lens from random galaxies, repeat until finds one
    which is an actual lens (i.e. supercritical)
    """
    
    default_kw_lenspart={"kwlens_part":kwlens_part_AS,
                         "projection_index":0,
                         "kw_prior_z_source":kw_prior_z_source_stnd,
                         "kwargs_band_sim":kwargs_band_sim,
                         "pixel_num":pixel_num,
                         "reload":reload}
    default_kw_lenspart.update(kw_lenspart)
    kw_lenspart = default_kw_lenspart
    default_kw_galpart={"simsuite":std_simsuite}
    default_kw_galpart.update(kw_galpart)
    kw_galpart = default_kw_galpart
    while True:
        Gal = get_rnd_PG(**kw_galpart)
        Gal.run()
        while kw_lenspart["projection_index"]<3:
            try:
                mod_LP = LensPart(Galaxy=Gal,
                              **kw_lenspart)
                mod_LP.run()
                return mod_LP            
            except ProjectionError as PE:
                kw_lenspart["projection_index"]+=1
        print("All projections of this galaxy are not supercritical #\n","Trying different galaxy")

            
# get ALL possible lenses
def wrapper_get_all_lens(reload=True,
                        kw_lenspart={},
                        kw_galpart={},
                        verbose=True):
    """Get a lens from all available galaxies"""
    
    default_kw_lenspart={"kwlens_part":kwlens_part_AS,
                         "projection_index":0,
                         "kw_prior_z_source":kw_prior_z_source_stnd,
                         "kwargs_band_sim":kwargs_band_sim,
                         "pixel_num":pixel_num,
                         "reload":reload}
    default_kw_lenspart.update(kw_lenspart)
    kw_lenspart = default_kw_lenspart
    default_kw_galpart={"simsuite":std_simsuite}
    default_kw_galpart.update(kw_galpart)
    kw_galpart = default_kw_galpart
    all_Gal    = get_all_PG(**kw_galpart)
    all_lenses = []
    if verbose:
        print(f"Found n={len(all_Gal)} Galaxies")
    for Gal in all_Gal:
        Gal.run(reload=reload)
        while kw_lenspart["projection_index"]<3:
            try:
                mod_LP = LensPart(Galaxy=Gal,
                              **kw_lenspart)
                mod_LP.run()
                all_lenses.append(mod_LP)
            except ProjectionError as PE:
                kw_lenspart["projection_index"]+=1
        print(f"All projections of Galaxy {Gal.name} are not supercritical \nTrying different galaxy.")
        kw_lenspart["projection_index"] = 0
        exit("DEBUG - check if this worked out")
    if verbose:
        print(f"Found n={len(all_lenses)} Lenses")
        print(f"i.e. {np.round(len(all_lenses)/len(all_Gal)*100,1)}% of Galaxies")
        
    return all_lenses