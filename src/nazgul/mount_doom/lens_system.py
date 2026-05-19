"""
Restructured from generate_particle_lens_dom with a different philosofy in mind:
This class should be as simple as possible, basically only a conveniency wrapper for LensModel, 
but geared toward the galaxy lens
"""
import numpy as np
from pathlib import Path
# for debug plots
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from scipy.interpolate import RectBivariateSpline

from lenstronomy.Util import util
from lenstronomy.Util.util import array2image,image2array 
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame

# My libs
from python_tools.tools import to_dimless,mkdir
import nazgul.mount_doom.cracks_of_doom as cod
from nazgul.project_gal import get_2Dkappa_map
from nazgul.mount_doom.generate_gal_lens import GalLens

# Default values
# particle lens class and params.
from nazgul.mount_doom.cracks_of_doom import source_model_list
from nazgul.configurations import pixel_num,min_thetaE
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from nazgul.mount_doom.cracks_of_doom import kwargs_band_sim,kw_prior_z_source_stnd

verbose = True
empty_kwargs_add_lenses = {"lens_model_list":[],"kwargs_lens":[]}


class LensSystem():
    def __init__(self,
                 kwargs_gallens,  #                 
                    #Galaxy,
                    #projection_index, # projection index of the galaxy 
                    #kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                    #pixel_num=pixel_num, # sim prms 
                    #kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                    #min_thetaE = min_thetaE,
                 kwargs_band_sim=kwargs_band_sim,
                 kwargs_add_lenses=empty_kwargs_add_lenses,
                 kwargs_lensmodel={}, 
                 ):
        """
        :param
            kwargs_gallens: Defining the Galaxy lens parameters:
                Galaxy :           PartGal instance
                kwlens_part:       kw_args of the PartLens class
                pixel_num:         int, number of pixel
                kw_prior_z_source: kwargs, define the prior distribution of the z_source to be sampled from
                min_thetaE:        float (ideally with units of arcsec), define the minumum thetaE for which the galaxy is considered a lens
            kwargs_band_sim: bandmodel used to define the Sim class
            kwargs_add_lenses: kwargs for additional lenses
            kwargs_lensmodel: additional kwargs to pass to LensModel (e.g. z_lens, as != z_galaxy)
        """ 

        # initialise it independently
        self.gallens            = GalLens(**kwargs_gallens)
        self.gallens.unpack()
        # LensModel
        self.kwargs_lensmodel    = kwargs_lensmodel
        self.kwargs_add_lenses   = kwargs_add_lenses
        # observational params
        self.kwargs_band_sim     = kwargs_band_sim
        # We assume that the cosmology has to be the same as the galaxy:
        self.cosmo               = self.gallens.cosmo
        mkdir(self.savedir)

    def setup(self,
                Sim=None,
                update_source_pos=False,
                verbose=verbose):
           
        self.gallens.unpack()
        self.gallens.run()
        self.create_lens(kwargs_add_lenses=self.kwargs_add_lenses, # these are given here as the self, but in principle with the freedom to re-define them
                         Sim=Sim,verbose=verbose)
        if update_source_pos:
            self.sample_source_pos(update=update_source_pos)
    #############################
    @property
    def savedir(self):
        return Path(self.gallens.Gal.gal_dir).parent/"LensSystem"
    @property
    def name(self):
        # define name and path of savefile
        name_lnsp = self.gallens.name
        # to correct asap
        name= name_lnsp.replace("Sub_","")
        return name
    @classmethod
    def from_GalLens(cls, GalLens,**kwargs_lenssystem):
        """Construct from an existing GalLens instance."""
        obj = cls.__new__(cls)
        obj.gallens = GalLens
        obj.gallens.unpack()
        # We assume the cosmology has to be the same as the Galaxy:
        obj.cosmo               = obj.gallens.cosmo        
        mkdir(obj.savedir)

        # LensModel
        obj.kwargs_lensmodel    = kwargs_lenssystem.get("kwargs_lensmodel",{})
        obj.kwargs_add_lenses   = kwargs_lenssystem.get("kwargs_add_lenses",empty_kwargs_add_lenses)
        # observational params
        obj.kwargs_band_sim     = kwargs_lenssystem.get("kwargs_band_sim",kwargs_band_sim)
        return obj
        
    #############################

    ###############
    ###############
    

    def create_lens(self,kwargs_add_lenses=empty_kwargs_add_lenses,Sim=None,verbose=verbose):
        # Define the radius based on ~ theta_E
        self.radius    = self.gallens.radius
        if verbose:
            print("Image radius:",np.round(self.radius,3))        
        # setup dataclasses (dataclass,psf_class,sourcemodel and some helper kwargs):
        self.setup_dataclasses(Sim=Sim,verbose=verbose)
        # setup lenses 
        self.setup_lenses(kwargs_add_lenses=kwargs_add_lenses,
                        verbose=verbose)

    def setup_dataclasses(self,Sim=None,verbose=True):
        """
        Define all classes not dependent on lensing
        Handled by SimAPI
        """
        if Sim is None:
            Sim = self.get_Sim()
        if verbose:
            print("Setting up data classes ...")
        self.data_class,self.psf_class,self.source_model_class,self.kwargs_numerics,self.kwargs_source = cod.get_dataclasses(Sim)
        if verbose:
            print("... Data classes set up")
        return 0

    def _reformat_kwargs_lensmodel(self):
        # ensure that kwargs_lensmodel has at least 
        # the default z_lens and z_source (if not input differently)
        default_kw_lm = {"z_lens":self.gallens.z_lens,
                         "z_source":self.gallens.z_source}
        
        kwargs_lensmodel = self.kwargs_lensmodel
        if kwargs_lensmodel is None or kwargs_lensmodel=={}:
            # for consistency, we set z_lens==z_gal and z_source == z_source_sampled
            kwargs_lensmodel     = default_kw_lm
        
        # update default_kw_lm with input kwargs_lensmodel if present
        default_kw_lm |= kwargs_lensmodel
        self.kwargs_lensmodel = default_kw_lm
        
    def setup_lenses(self,kwargs_add_lenses=empty_kwargs_add_lenses,verbose=verbose):
        """
        Setup lensing parameters tailored for lenstronomy
        """
        if hasattr(self,"lens_model") and hasattr(self,"kwargs_lens"):
            if verbose:
                print("Lens model already setup")
            return 0
        if verbose:
            print("Setting up lensing parameters...")
        add_lens_model_list    = kwargs_add_lenses["lens_model_list"]
        add_kwargs_lens        = kwargs_add_lenses["kwargs_lens"]
        lens_model_list        = ["PART_GAL",*add_lens_model_list]
        self.kwargs_lens       = [{},*add_kwargs_lens]
        pkwl_part_lens         = {"lenspart":self.gallens}
        self._reformat_kwargs_lensmodel()
        # the following in principle it is not necessary anymore...?
        if getattr(self.kwargs_lensmodel,"z_lens",self.gallens.z_lens)!=self.gallens.z_lens:
            pkwl_part_lens["z_lens"] = self.kwargs_lensmodel["z_lens"]
        if getattr(self.kwargs_lensmodel,"z_source",self.gallens.z_source)!=self.gallens.z_source:
            pkwl_part_lens["z_source"] = self.kwargs_lensmodel["z_source"]
        profile_kwargs_list    = [pkwl_part_lens]
        for adlml in add_lens_model_list:
            profile_kwargs_list.append({})
        self.lens_model = LensModel(lens_model_list=lens_model_list,
                                    profile_kwargs_list = profile_kwargs_list,
                                    **self.kwargs_lensmodel)
        if verbose:
            print("... Lensing parameters set up")
    
    ########################
    # Lensing Computations #
    ########################
    
    def alpha_map(self,_radec=None):
        if self.kwargs_add_lenses == empty_kwargs_add_lenses:
            return self.gallens._alpha_map(_radec=_radec)
        if _radec is None:
            _radec = self.gallens._radec
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_model.alpha(_ra, _dec, self.kwargs_lens)
        alpha_x,alpha_y = array2image(alpha_x),array2image(alpha_y)
        return alpha_x,alpha_y
        
    def psi_map(self,_radec=None):
        print("Computing lensing PM potential...")
        if self.kwargs_add_lenses == empty_kwargs_add_lenses:
            return self.gallens._psi_map(_radec=_radec)
        if _radec is None:
            _radec = self.gallens._radec #arcsecs  
        _ra,_dec = _radec
        psi = self.lens_model.potential(_ra, _dec, self.kwargs_lens)
        psi = array2image(psi)
        return psi
        
    def _kappa_map_from_lens(self,_radec=None,exact=False):
        # compute analytically from the particles -> actually should not be the way to do it !
        print("Computing kappa map from PM...")
        # the following should prob. be in the unpack function
        self.gallens.setup_lenses()
        if _radec is None:
            _radec = self.gallens._radec
        _ra,_dec = _radec
        kappa = self.lens_model.kappa(_ra, _dec, self.kwargs_lens)
        kappa = array2image(kappa)
        return kappa
        
    def kappa_map(self,_radec=None):
        # compute from density map
        # actually better bc does not depend on the particle profile
        print("Computing kappa map from density map...")
        kappa = self.gallens._kappa_map(_radec=_radec)
        if self.kwargs_add_lenses["lens_model_list"]!=[]:
            # Add kappa of the additional profiles
            if _radec is None:
                _radec = self.gallens._radec
            _ra,_dec   = _radec
            lens_model_only_add = LensModel(lens_model_list=self.kwargs_add_lenses["lens_model_list"],
                                           **self.kwargs_lensmodel)
            kappa_add  = lens_model_only_add.kappa(_ra, _dec, self.kwargs_add_lenses["kwargs_lens"])
            kappa     += array2image(kappa_add)
        return kappa
            
    # Shear components and caustics/CL
    def hessian(self,_radec=None):
        """Computes the hessian matrix on the grid by taking the gradient 
        of the alpha map
        """
        print("Computing hessian as gradient of the deflection map...")
        # Can be now computed also beyond the cutout
        if self.kwargs_add_lenses == empty_kwargs_add_lenses:
            return self.gallens._hessian(_radec=_radec)
        if _radec is None:
            _radec          = self.gallens._radec
        alpha_x,alpha_y = self.alpha_map(_radec=_radec)
        _ra,_dec = _radec
        RA0,DEC0 = array2image(_ra)[0],array2image(_dec)[:,0]
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, RA0,DEC0)
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, RA0,DEC0)
        f_xx,f_xy,f_yx,f_yy  = dalpha_x_dx,dalpha_x_dy,dalpha_y_dx,dalpha_y_dy
        return f_xx,f_xy,f_yx,f_yy

    def get_kw_shear(self,_radec=None):
        """From the hessian matrix compute the shear components
        """
        f_xx,f_xy,f_yx,f_yy = self.hessian(_radec=_radec)
        # derived shear1,shear2 and shear
        shear1 = 1./2 * (f_xx - f_yy)
        shear2 = f_xy
        shear  = np.hypot(shear1,shear2)
        kw_shear = {"shear1":shear1,"shear2":shear2,"shear":shear}
        return kw_shear
        
    def shear_map(self,_radec=None):
        return self.get_kw_shear(_radec=_radec)["shear"]

    # Caustics 
    ##########
    
    def get_kw_critical_curve_caustics(self,
                                       _radec=None):
        """ Fit the critical curve and map it to the caustic
        """
        # note: alpha is computed from the particle (and shear from alpha)
        # thus depends on the particle lens model chosen, while kappa
        # is obtained directly as a density map + cosmological scaling
        alpha_x,alpha_y = self.alpha_map(_radec=_radec)
        kappa           = self.kappa_map(_radec=_radec)
        shear           = self.shear_map(_radec=_radec)

        # in principle, the _radec would give the number of pixel
        if _radec:
            pixel_num = int(np.sqrt(_radec[0].shape))
        else:
            pixel_num = self.gallens.pixel_num
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
            if len(ierx)/(pixel_num**2) >0.001 :
                break
            else:
                iery,ierx = np.where(cod.MAD_mask(np.abs(eigen_rad),0,tv)) 
        # tangential
        mintv = np.min(np.abs(eigen_tan))
        Dv = np.max(np.abs(eigen_tan)) - mintv
        maxtv = mintv + 0.1*Dv
        test_values = np.linspace(mintv,maxtv,20)
        for tv in test_values:
            if len(ietx)/(pixel_num**2) >0.001:
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
        if _radec is None:
            _radec          = self.gallens._radec
        RA,DEC      = cod.get_RADEC(_radec)
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
    ########################
    ########################
    
    ########################
    # Creating Lensed image:
    ########################
    
    # Source
    #########
    def update_source_position(self,ra_source,dec_source):
        # useful if we want to put it in the center of the caustic
        self.kwargs_source["center_x"] = ra_source
        self.kwargs_source["center_y"] = dec_source
        return 0
        
    def sample_source_pos(self,update=False,_radec=None):
        """Sample the source position within the tangential
        critical caustic
        TODO: optimise this computation and/or store the results
        """
        print("Sampling source position within tangential caustic")
        if _radec is None:
            _radec = self.gallens._radec
        kw_caustics  = self.get_kw_critical_curve_caustics(_radec=_radec)
        ra_ct,dec_ct = kw_caustics["caustics"]["tangential"]
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

    def get_xy_source_plane(self,alpha_map=None,_radec=None):
        """Map the x,y grid into the source plane
        (used to fit the light of the source to the image)
        """
        # coords
        if _radec is None:
            _radec  = self.gallens._radec
        RA,DEC      = cod.get_RADEC(_radec)
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
        
    def get_lensed_image(self,
                         Sim=None,
                         _radec=None, # 
                         unconvolved=True):
        
        imageNumerics,sourceModel = self.get_imageNumerics(Sim=Sim,return_sourceModel=True)
        kwargs_source = imageNumerics.kwargs_source
        alpha_map = self.alpha_map(_radec=_radec)
        x_source_plane,y_source_plane = self.get_xy_source_plane(alpha_map=alpha_map)
        kwargs_source_list            = [kwargs_source]
        source_light                  = sourceModel.surface_brightness(x_source_plane, y_source_plane, kwargs_source_list, k=None)
        # the following is to compare the Sim to the "native" resolution of the gallens image
        sim_deltapix                  = imageNumerics.grid_class.pixel_width
        if not np.abs(sim_deltapix-to_dimless(self.gallens.deltaPix))<1e-10:
            if sim_deltapix<to_dimless(self.gallens.deltaPix):
                raise RuntimeError("Simulated observatory cannot have higher resolution than baseline simulation") 
            # different pixel scale (due to different simulated observatory)
            source_light_im        = array2image(source_light)
            pixel_num_aim          = imageNumerics.grid_class.num_grid_points_axes[0]
            source_light_downgrade = zoom(source_light_im,pixel_num_aim/self.gallens.pixel_num)
            source_light           = image2array(source_light_downgrade)
        image_sim = imageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return image_sim

    # Simulating observations
    #########################
    
    def get_imageNumerics(self,Sim=None,return_sourceModel=False):    
        if Sim is None:
            Sim = self.get_Sim()
            if not hasattr(self,"data_class"):
                self.setup_dataclasses(Sim=Sim)
            data_class      = self.data_class
            psf_class       = self.psf_class
            sourceModel     = self.source_model_class 
            kwargs_source   = self.kwargs_source
            kwargs_numerics = self.kwargs_numerics     
        else:
            data_class,psf_class,sourceModel,kwargs_numerics,kwargs_source = cod.get_dataclasses(Sim)
        imageNumerics = self._get_imageNumerics(data_class=data_class,
                                                psf_class=psf_class,
                                                kwargs_numerics=kwargs_numerics)
                                                
        # add this for convenience:
        imageNumerics.kwargs_source = kwargs_source
        if return_sourceModel:
            return imageNumerics,sourceModel
        return imageNumerics
                                        
    def _get_imageNumerics(self,data_class,psf_class,kwargs_numerics):
        imageNumerics = NumericsSubFrame(pixel_grid=data_class,
                                         psf=psf_class, 
                            **kwargs_numerics)
        return imageNumerics
 
    
    def get_Sim(self,band=None,
                    kwargs_psf=None,# add psf and pssf
                    source_model_list=source_model_list):
        kwargs_source_model = {"source_light_model_list":source_model_list,
                                   "cosmo":self.cosmo
                                   }
        kwargs_model = {"z_source":self.gallens.z_source} | kwargs_source_model
        if not "cosmo" in kwargs_model.keys():
            # cosmology should in principle not be used, but better be consistent
            kwargs_model["cosmo"] = self.cosmo
        if band is None:
            # This is the resolution of the simulation itself
            kwargs_single_band = self.kwargs_band_sim
            kwargs_single_band["pixel_scale"] = to_dimless(self.gallens.deltaPix)
            pixel_num = self.gallens.pixel_num
        else:
            kwargs_single_band = band.kwargs_single_band()
            if kwargs_psf is not None:
                if not kwargs_psf.keys() == {"kernel_point_source":[],
                                             "point_source_supersampling_factor":[]}.keys():
                    raise RuntimeError(f"kwargs_psf has to have only kernel_point_source and point_source_supersampling_factor, not {kwargs_psf.keys()}")
                kwargs_single_band.update(kwargs_psf)
            # must recompute pixel_num in order to covert to ~ the same aperture,
            # but with the new resolution 
            # -> round down to be sure we are within the bounds
            pixel_num = int(to_dimless(2*self.gallens.radius)/kwargs_single_band["pixel_scale"])
        # instantiate simulation API class
        Sim = SimAPI(numpix = pixel_num, # N of pixels in "observed" image
                 kwargs_single_band = kwargs_single_band, # telescope specific keyword arguments (eg HST, see above)
                 kwargs_model = kwargs_model,# kwargs source model (in principle kw lens as well)
                )
        return Sim

    def sim_image(self,SimObs,noisy=False):
        """Obtain simulated images given SimObs
        """
        image_SimObs     = self.get_lensed_image(Sim=SimObs,
                                             unconvolved=False)
        if noisy:
            image_SimObsnoisy  = image_SimObs + SimObs.noise_for_model(model=image_SimObs)
            error_SimObs       = SimObs.estimate_noise(image_SimObsnoisy) # NOT variance
            return image_SimObsnoisy,error_SimObs
        return image_SimObs

    def sim_multi_band_list(self,band,kwargs_psf=None,source_model_list=source_model_list):
        """Setup Simulation given band specific, its psf and source_model_list 
        """
        SimObs                         = self.get_Sim(band=band,kwargs_psf=kwargs_psf,
                                                         source_model_list=source_model_list)
        image_SimObsnoisy,error_SimObs = self.sim_image(SimObs,noisy=True)
        kw_data_sim               = SimObs.kwargs_data
        kw_data_sim["image_data"] = image_SimObsnoisy
        kw_data_sim["noise_map"]  = error_SimObs
        kw_psf_sim                = SimObs.kwargs_psf
        # consider if to add psf error - depends on observations
        if "point_source_supersampling_factor" in kw_psf_sim.keys():
            kw_numerics_sim = {'point_source_supersampling_factor':kw_psf_sim["point_source_supersampling_factor"]} 
        else:
            kw_numerics_sim = {'point_source_supersampling_factor':1}
        image_band      = [kw_data_sim, kw_psf_sim, kw_numerics_sim]
        multi_band_list = [image_band]
        return multi_band_list
    
