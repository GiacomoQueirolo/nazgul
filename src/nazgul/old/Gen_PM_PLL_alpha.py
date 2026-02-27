# copy from Gen_PM_PLL 5/12/25
# -> implement minor structural changes to save the data
# -> implement major struct. changes regarding image/alpha:
#         -> not compute "directly" the lensed image, rather the alpha map -> save the alpha maps
#            and recover then the image afterwards -> don't save the image
#          -> this will allow us to 1) compute the approx caustics -> implemented here
#                                   2) later add the LOS effects on the alpha map directly

import os,pickle
from scipy.stats import norm
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt
from functools import cached_property 
from mpl_toolkits.axes_grid1 import make_axes_locatable

from concurrent.futures import ThreadPoolExecutor

from lenstronomy.Util import util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from LensModel.lens_model import LensModel # modified!!
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

import astropy.units as u
import astropy.constants as const



from lenstronomy.Plots import plot_util

from python_tools.fwhm import get_fwhm
from python_tools.tools import mkdir,short_SciNot,to_dimless



pixel_num     = 200 # pix for image
pixel_dens    = 100 # pixel for mass density
verbose       = True
# for z_source computation:
z_source_max  = 4
# cutoff to define the 2D density hist
# from there the maximum density pixel
# and from there min_z_source
cutoff_radius = 100*u.kpc 



# cosmol. params.
from lib_cosmo import SigCrit
##########################
##########################
# Sampling of the Profiles
##########################
##########################
#
# Particle functions
#
# cosmo from https://academic.oup.com/mnras/article/474/3/3391/4644836, Agnello 2017
# point mass theta_E (from eq.4.7 of Meneghetti's lecture note - and by memory)
# theta_E = \sqrt ( 4GM D_ls / (c^2 Ds Dl) )
# divide the computation such that it's done only once
def thetaE_PM_prefact(z_lens,z_source,cosmo):    
    cosmo_ds  = cosmo.angular_diameter_distance(z_source)
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens)
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source)
    pref      = 4*const.G*cosmo_dds/(const.c*const.c*cosmo_ds*cosmo_dd)
    return np.sqrt(pref) # 

@u.quantity_input
def thetaE_PM(M:u.g,theta_pref:u.g**-.5):
    thetaE_rad = np.sqrt(M)*theta_pref
    thetaE     = thetaE_rad.to("")*u.rad.to("arcsec")
    return thetaE.value #in arcsec
# ARCSINH thetaE is actually the same as PM
def thetaE_AS_prefact(z_lens,z_source,cosmo):    
    # is actually the same of PM, but it could be in principle different
    return thetaE_PM_prefact(z_lens,z_source,cosmo)
@u.quantity_input
def thetaE_AS(M:u.g,theta_pref:u.g**-.5):
    # is actually the same of PM, but it could be in principle different
    return thetaE_PM(M,theta_pref)

# maybe useful funct:
def MfromtE(tE,theta_pref:u.g**-.5):
    tErad  = tE*u.arcsec.to("rad")
    M = (tErad/theta_pref)**2
    return M.to("Msun")

def _build_kwargs_lens_AS(args):
        tE,tCAS, ra, dec = args
        return {
            "theta_E": tE,
            "theta_c": tCAS,
            "center_x": ra,
            "center_y": dec
        }
        
def _build_kwargs_lens_PM(args):
        tE, ra, dec = args
        return {
            "theta_E": tE,
            "center_x": ra,
            "center_y": dec
        }


# From EAGLE simulation

from remade_gal import get_rnd_NG,get_lens_dir
from project_gal_alpha import prep_Gal_projpath
from project_gal_alpha import Gal2kw_samples,projection_main,project_kw_parts,kwparts2arcsec
# Helper funct - create the kwargs_lens given the part. parameters, ie theta_E,x,y,(core if needed) 
# and the lens_model
#
def get_lens_model_PM(thetaEs,samples):
    kwargs_lens_PM  = [_build_kwargs_lens_PM((thetaEs, samples[0],samples[1]))]
    lens_model_list = ["POINT_MASS"]
    lens_model_PM   = LensModel(lens_model_list=lens_model_list)
    return kwargs_lens_PM,lens_model_PM 

def get_lens_model_AS(theta_cAS,thetaEs,samples):
    try:
        len(theta_cAS)
    except TypeError:
        theta_cAS *= np.ones_like(thetaEs)
    kwargs_lens_AS = [_build_kwargs_lens_AS((thetaEs,theta_cAS, samples[0],samples[1]))]
    lens_model_list = ["ARSINH_PARALL"]
    lens_model_AS   = LensModel(lens_model_list=lens_model_list)
    return kwargs_lens_AS,lens_model_AS
    
#
# Particle functions
# -> not needed anymore due to structural changes
    
"""
def get_kwrg_PM(samples,Ms,
                    z_lens,z_source,
                    theta_E):     
    theta_pref = thetaE_PM_prefact(z_lens=z_lens,z_source=z_source)
    thetaEs    = thetaE_PM(M=Ms,theta_pref = theta_pref)

    kwargs_lens_PM,lens_model_PM = get_lens_model_PM(thetaEs,samples)
    return {"kwargs_lens_PART":kwargs_lens_PM,"lens_model_PART":lens_model_PM}
                
def get_kwrg_AS(samples,Ms,theta_cAS
                    z_lens,z_source,
                    theta_E):
                    
    theta_pref = thetaE_AS_prefact(z_lens=z_lens,z_source=z_source)
    thetaEs    = thetaE_AS(M=Ms,theta_pref=theta_pref)
 
    kwargs_lens_AS,lens_model_AS = get_lens_model_AS(theta_cAS,thetaEs,samples)

    return {"kwargs_lens_PART":kwargs_lens_AS,"lens_model_PART":lens_model_AS}
"""
#
# naming functions
#

def _get_tcAS_str(kwargs_lens):
    try:
        tcAS = kwargs_lens["theta_cAS"].value
    except AttributeError:
        tcAS = kwargs_lens["theta_cAS"]
    tcAS_str = short_SciNot(tcAS)
    return tcAS_str 
    
def get_name_PM(kw_lens=None):
    return f"PM"
def get_name_AS(kwargs_lens):
    tcAS_str   = _get_tcAS_str(kwargs_lens)
    return f"AS_tc{tcAS_str}"

# Lens modelling 
#################

#
# Class wrapper for Particle Lens computation
#
class PMLens():
    def __init__(self,kwargs_lens_part):
        self.kwargs_lens = kwargs_lens_part
        type_part = kwargs_lens_part["type"]
        self.name = type_part
        
        if type_part=="PM":
            self.thetaE_prefact = thetaE_PM_prefact
            self.thetaE         = thetaE_PM
            self.get_lens_model = get_lens_model_PM

        elif type_part=="ARCSINH" or type_part=="AS":
            self.thetaE_prefact = thetaE_AS_prefact
            self.thetaE         = thetaE_AS
            self.get_lens_model = get_lens_model_AS 
        else:
            raise TypeError("This particle model is not known: "+type_part)

    
    def setup(self,Mod):
         self.z_lens   = Mod.z_lens
         self.z_source = Mod.z_source
         self.cosmo    = Mod.cosmo
                                          
    def get_lens_PART(self,samples,Ms):
        theta_pref = self.thetaE_prefact(z_lens=self.z_lens,z_source=self.z_source,cosmo=self.cosmo)
        thetaEs    = self.thetaE(M=Ms,theta_pref = theta_pref)
        kw_lns_mod = {}
        if self.name =="ARSINH"  or self.name =="AS":
            kw_lns_mod = {"theta_cAS":self.kwargs_lens["theta_cAS"]}
        kwargs_lens_PART,lens_model_PART = self.get_lens_model(thetaEs=thetaEs,samples=samples,**kw_lns_mod)
        return kwargs_lens_PART,lens_model_PART

    
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        # convert kwargs in immuatable tuple to be hashable
        Id = (self.name,
              tuple(sorted(self.kwargs_lens.items())))
        return Id
    
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, PMLens):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        if not getattr(self,"name",False):
            self._setup_names()
        return self.name
    ########################
    ########################
"""
    def __eq__(self,other):
        if not hasattr(other,"name"):
            return False
        if other.name!=self.name:
            return False
        if not hasattr(other,"kwargs_lens"):
            return False
        if not self.kwargs_lens.keys()==other.kwargs_lens.keys():
            return False
        for k in self.kwargs_lens.keys():
            if self.kwargs_lens[k]!=other.kwargs_lens[k]:
                return False
        return True
"""
def get_LensSystem_kwrgs(deltaPix,pixel_num=pixel_num,background_rms=0.005,exp_time=500,ra_source=0.,dec_source = 0.):
    # data specifics
    # background noise per pixel
    # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    print("Pixel_num: ",  pixel_num)
    print("DeltaPix: ",  np.round(deltaPix,3))
    deltaPix = to_dimless(deltaPix,True) # if dimensional, convert to dimensionless
    kwargs_data = sim_util.data_configure_simple(pixel_num, deltaPix, exp_time, background_rms)
    data_class  = ImageData(**kwargs_data)
    kwargs_psf  = {'psf_type': 'NONE'}  
    psf_class   = PSF(**kwargs_psf)
    # Source Params
    source_model_class,kwargs_source = get_model_source(ra_source,dec_source)
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # for modelling later:
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 
                     'multi_band_type': 'single-band'}
    return data_class, psf_class, source_model_class, kwargs_numerics, kwargs_source,kwargs_data_joint

def get_model_source(ra_source=0,dec_source=0):
    source_model_list = ['SERSIC_ELLIPSE']
    
    kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 
                            'center_x': ra_source,
                             'center_y': dec_source, 
                             'e1': -0.1, 'e2': 0.01}

    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    return source_model_class,kwargs_source
    
##########################
# Model class for parts. #
##########################
theta_c_AS     = 5e-3 
kwlens_part_AS = {"type":"AS","theta_cAS":theta_c_AS}
kwlens_part_PM = {"type":"PM"}
from time import time # for DEBUG
from remade_gal import Gal2kwMXYZ
from project_gal_alpha import get_rough_radius


class LensPart(): 
    def __init__(self,
                    Galaxy,
                    kwlens_part, # if PM or AS, and if so size of the core
                    pixel_num=pixel_num, # sim prms 
                    cutoff_radius = cutoff_radius, # ONLY for z_source computation (!=! radius, which is instead obtained from an estimate of theta_E)
                    z_source_max  = z_source_max,     # for z_source sampling
                    #z_lens=z_lens,z_source=z_source,cosmo=cosmo, # cosmo prms -> obtained from Galaxy 
                    exp_time=500,bckg_rms=0.01, # observation parameters
                    savedir_sim="lensing",reload=True # saving params
                    ):
        Galaxy            = prep_Gal_projpath(Galaxy) # just set up some directories
        # setup of data
        self.Gal           = Galaxy
        self.kwlens_part   = kwlens_part
        lens_dir           = get_lens_dir(self.Gal)
        self.savedir_sim   = savedir_sim
        mkdir(savedir_sim)
        self.savedir       = f"{lens_dir}/{savedir_sim}"
        self.reload        = reload
        self.pixel_num     = pixel_num      
        mkdir(self.savedir)
        # cosmo prms
        self.z_lens        = self.Gal.z
        self.cosmo         = self.Gal.cosmo
        self.arcXkpc       = self.cosmo.arcsec_per_kpc_proper(self.z_lens)

        # To obtain the z_source and projection index 
        #-> computed only once in the run function
        self.z_source_max  = z_source_max
        self.cutoff_radius = cutoff_radius
        
        # observational params
        self.exp_time  =  exp_time
        self.bckg_rms  =  bckg_rms
        # kwargs_lens for the particles :
        # type:"AS" or "PM"
        # if AS: param:{"thetacAS"}
        self.PMLens      = PMLens(kwlens_part)
        self._setup_names()

    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self.Gal._identity(),
              self.PMLens._identity(),
              self.pixel_num,
              self.z_source_max,self.cutoff_radius,
              self.exp_time,self.bckg_rms)
        return Id
    
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, LensPart):
            return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        if not getattr(self,"name",False):
            self._setup_names()
        return self.name
    # the following struct. is more clear and allow a slimmer stored class
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove large but recomputable attributes (if present)
        state.pop('kwargs_lens_PART', None)
        state.pop('kw_shear',None)
        # could also remove image_sim if it's heavy
        try:
            state['imageModel'].ImageNumerics._numerics_subframe._grid = None
        except Exception:
            pass
        return state

    def __setstate__(self, state):
        # Optional: restore defaults or trigger rebuild of heavy attributes
        self.__dict__.update(state)

    def store(self):
        if not hasattr(self,"pkl_path"):
            self._setup_names()
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self, f)
        print("Saved", self.pkl_path)
        
    def unpack(self):
        # this function recover the parts deleted before storing
        # to save space
        # recover the grid
        gridClassType = getattr(self.kwargs_numerics,"compute_mode","regular")
        if gridClassType =="regular":
            from lenstronomy.ImSim.Numerics.grid import RegularGrid as Grid
        elif gridClassType  == "adaptive":
            from lenstronomy.ImSim.Numerics.grid import AdaptiveGrid as Grid
        recomputed_grid = Grid(nx=self.pixel_num,ny=self.pixel_num,
                        transform_pix2angle=self.imageModel.Data.transform_pix2angle,
                        ra_at_xy_0=self.imageModel.Data.radec_at_xy_0[0],
                        dec_at_xy_0=self.imageModel.Data.radec_at_xy_0[1])
        self.imageModel.ImageNumerics._numerics_subframe._grid = recomputed_grid 
        # recover also kwargs_lens_PART:
        self.setup_lenses() 
        return True
    ########################
    ########################

    def _get_name(self):
        # define name and path of savefile
        self.name       = f"{self.Gal.Name}_Npix{self.pixel_num}_Part{self.PMLens.name}"
        
    def _setup_names(self):
        if not getattr(self,"name",False):
            self._get_name()
        self.pkl_path = f"{self.savedir}/{self.name}.pkl"
        
    def upload_prev(self):
        if not self.reload:
            return False
        prev_mod = ReadLens(self)
        if prev_mod is False:
            return False
        # we have now a good way to define equality
        if prev_mod==self:
            for attr, value in prev_mod.__dict__.items():
                setattr(self, attr, value)
            return True
        return False
            
    def run(self,read_prev=True):
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            # Read particles ONLY ONCE
            kw_parts          = Gal2kwMXYZ(self.Gal) # kwargs of Msun,XYZ in kpc (explicitely) centered around Centre of Mass (CM)
            
            kwres_proj = projection_main(Gal=self.Gal,kw_parts=kw_parts,
                            pixel_num=pixel_dens,z_source_max=self.z_source_max,
                            arcXkpc=self.arcXkpc,verbose=True,save_res=True,reload=True)

            self.proj_index   = kwres_proj["proj_index"]
            self.z_source_min = kwres_proj["z_source_min"]
            print("Min Z source:",self.z_source_min)
            print("Max Z source:",self.z_source_max)
            self.z_source     = self.sample_z_source(z_source_min=self.z_source_min,
                                                     z_source_max=self.z_source_max)
            print("Z source sampled:",self.z_source)
            # the following 2 can only be computed once we know the z_source:
            self.SigCrit       = SigCrit(cosmo=self.cosmo,z_lens=self.z_lens,z_source=self.z_source) # Msun/kpc^2

            self.PMLens.setup(self) # only run now bc it needs z_source 
            
            kw_parts_proj  = project_kw_parts(kw_parts,self.proj_index)
            kw_part_arc    = kwparts2arcsec(kw_parts_proj,self.arcXkpc)
            # Then define the radius based on ~ theta_E
            # note: computation actually pretty fast for get_rough_radius (~0.1sec)
            self.radius    = get_rough_radius(cosmo=self.cosmo,
                                                z_lens=self.z_lens,
                                                z_source=self.z_source,
                                                kw_part_arc=kw_part_arc,kw_2Ddens=kwres_proj,
                                                scale=2,verbose=True) # [arcsec]
            print("Image radius:",np.round(self.radius,3))
    
            Diam_arcsec      = 2*self.radius #diameter in arcsec
            self.deltaPix    = Diam_arcsec/self.pixel_num # ''/pix

            # images:
            self.setup_dataclasses()
            self.setup_lenses()
            # this is the most computationally intense function:
            self.image_sim  = self.get_lensed_image()
            self.store()

    def setup_dataclasses(self,ra_source=0,dec_source=0):
        self.data_class,self.psf_class,self.source_model_class,\
        self.kwargs_numerics, self.kwargs_source,self.kwargs_data_joint = \
                get_LensSystem_kwrgs(self.deltaPix,self.pixel_num,background_rms=self.bckg_rms,exp_time=self.exp_time,
                                     ra_source =ra_source,dec_source =dec_source)
        return 0
    def update_source_position(self,ra_source,dec_source):
        # useful if we want to put it in the center of the caustic
        self.source_model_class,self.kwargs_source = get_model_source(ra_source,dec_source)
        
    # the following is meant to be rerun every time we load the class to save space
    # -> computationally not too intense (medium tho)
    def setup_lenses(self):
        print("Setting up lensing parameters")
        # Convert x,y,z in samples and get masses
        kw_samples = Gal2kw_samples(self.Gal,self.radius,self.proj_index,self.arcXkpc)
        samples    = kw_samples["RAs"],kw_samples["DECs"]
        Ms         = kw_samples["Ms"]
        # Convert in lenses parameters 
        kwLns_PART,LnsMod_PART = self.PMLens.get_lens_PART(samples=samples,Ms=Ms)
        self.kwargs_lens_PART  = kwLns_PART
        self.lens_model_PART   = LnsMod_PART
        return 0
        
    def sample_z_source(self,z_source_min,z_source_max):
        # this is here to allow modularity 
        # for now a simple uniform sample, but we could define something more fancy
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        return z_source
        
    def get_lensed_image(self):
        sourceModel  = self.source_model_class
        imageModel   = self.imageModel
        RA,DEC       = self.get_RADEC()
        # if not already, compute alpha_map
        alpha_x,alpha_y = self.alpha_map
        print("Centering source in the center of the tangential caustic")
        kw_caustics = self.critical_curve_caustics()
        ra_ct,dec_ct = kw_caustics["caustics"]["tangential"]
        self.update_source_position(np.mean(ra_ct),np.mean(dec_ct))
        
        x_source, y_source = RA-alpha_x,DEC-alpha_y
        # the coords have to be given as flat
        x_source = util.image2array(x_source)
        y_source = util.image2array(y_source)
        source_light = sourceModel.surface_brightness(x_source, y_source, self.kwargs_source, k=None)
        image_sim  = imageModel.ImageNumerics.re_size_convolve(source_light, unconvolved=False)
        # imageModel.image(self.kwargs_lens_PART, self.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
        return image_sim
        
    @cached_property
    def imageModel(self):
        if not hasattr(self,"setup_dataclasses"):
            self.setup_dataclasses()
        if not hasattr(self,"lens_model_PART"):
            self.setup_lenses()
        imageModel = ImageModel(self.data_class, self.psf_class, 
                        self.lens_model_PART, 
                        self.source_model_class,lens_light_model_class=None,point_source_class=None, 
                        kwargs_numerics=self.kwargs_numerics)
        return imageModel
    
    @cached_property
    def alpha_map(self):
        return self._alpha_map(_radec=None)
    @cached_property
    def kappa_map(self):
        return self._kappa_map(_radec=None)
        
    def _alpha_map(self,_radec=None):
        print("Computing lensing PM deflection...")
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        alpha_x,alpha_y = self.lens_model_PART.alpha(_ra, _dec, self.kwargs_lens_PART)
        alpha_x,alpha_y = util.array2image(alpha_x),util.array2image(alpha_y)
        return alpha_x,alpha_y
        
    def _kappa_map(self,_radec=None):
        # very important! this is kappa only within the radec considered
        print("Computing kappa map from PM...")
        if _radec is None:
            _radec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        _ra,_dec = _radec
        kappa = self.lens_model_PART.kappa(_ra, _dec, self.kwargs_lens_PART)
        kappa = util.array2image(kappa)
        return kappa

    def get_RADEC(self):
        _ra,_dec = self.imageModel.ImageNumerics.coordinates_evaluate #arcsecs  
        RA,DEC   = util.array2image(_ra),util.array2image(_dec)
        return RA,DEC

    def update_kwargs_data_joint(self,add_noise=False):
        # updating it with the PM image
        if add_noise:
            image   = self.image_sim
            poisson = image_util.add_poisson(image, exp_time=self.exp_time)
            bkg     = image_util.add_background(image, sigma_bkd=self.bckg_rms)
            
            self.image_sim = image + poisson + bkg

            #other realisation of the bkg
            bkg       = image_util.add_background(image, sigma_bkd=self.bckg_rms)
            noise_map =  np.sqrt(poisson**2+bkg**2)
            self.kwargs_data_joint["multi_band_list"][0][0]["noise_map"] = noise_map
        self.kwargs_data_joint["multi_band_list"][0][0]["image_data"] = self.image_sim

    # Shear components and caustics/CL
    @cached_property    
    def hessians(self):
        alpha_x,alpha_y = self.alpha_map
        # taking the non-dimensional pixel scale for the gradient
        dalpha_x_dy, dalpha_x_dx = np.gradient(alpha_x, to_dimless(self.deltaPix))
        dalpha_y_dy, dalpha_y_dx = np.gradient(alpha_y, to_dimless(self.deltaPix))
        print("Note: Taking the average of dalpha_x_dy and dalpha_y_dx for fxy")
        f_xx,f_xy,f_yy  = dalpha_x_dx,(dalpha_x_dy+dalpha_y_dx)/2.,dalpha_y_dy
        return f_xx,f_xy,f_yy
        
    def get_kw_shear(self):
        f_xx,f_xy,f_yy = self.hessian()
        # derived kappa, shear1,shear2 and shear
        kappa  = (f_xx + f_yy)/2.
        np.testing.assert_almost_equal(kappa,self.kappa_map,decimal=2)
        shear1 = 1./2 * (f_xx - f_yy)
        shear2 = f_xy
        shear  = np.hypot(shear1,shear2)
        self.kw_shear = {"shear1":shear1,"shear2":shear2,"shear":shear}
        return self.kw_shear
        
    @property
    def shear_map(self):
        if hasattr(self,"kw_shear"):
            return self.kw_shear["shear"]
        return self.get_kw_shear["shear"]
        
    def critical_curve_caustics(self):
        # no need to save this - as long as we have {kappa,shear,alpha}_map,
        # it's fast to compute
        kappa = self.kappa_map
        shear = self.shear_map
        alpha_x,alpha_y = self.alpha_map
        # radial and tangential eigenvalues
        eigen_rad = 1 - kappa + shear
        eigen_tan = 1 - kappa - shear
        # have to find when those are ~0
        mask_rad = MAD_mask(eigen_rad)
        mask_tan = MAD_mask(eigen_tan)
        ierx,iery = np.where(mask_rad.T==1)
        ietx,iety = np.where(mask_tan.T==1)
        # coords
        RA,DEC      = self.get_RADEC()
        ra0,dec0    = RA[0],DEC.T[0]
        # critical and caustics divided in tangential and radial
        cl_rad_x,cl_rad_y   = ra0[ierx],dec0[iery]    
        cc_rad_x,cc_rad_y   = ra0[ierx]-alpha_x[iery,ierx],dec0[iery]-alpha_y[iery,ierx]  
        cl_tan_x,cl_tan_y   = ra0[ietx],dec0[iety]    
        cc_tan_x,cc_tan_y   = ra0[ietx]-alpha_x[iety,ietx],dec0[iety]-alpha_y[iety,ietx]  
        kw_crit = {"caustics":{"radial":[cc_rad_x,cc_rad_y],"tangential":[cc_tan_x,cc_tan_y]},
                   "critical_lines":{"radial":[cl_rad_x,cl_rad_y],"tangential":[cl_tan_x,cl_tan_y]}
                  }
        return kw_crit
                
def MAD_mask(values,v0=0,sigma_scale=3):
    # robust estimator of noise: Median Absolute Deviation    
    mad = np.median(np.abs(values - np.median(values)))

    sigma = 1.4826 * mad

    mask = np.abs(values-v0) < sigma_scale*sigma   # ~99.7% Gaussian confidence
    return mask



#
# helper funct
#

# this function is a wrapper for convenience - it takes the class itself as input
from python_tools.get_res import LoadClass

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




def _plot_caustics(LPClass,
                   lensModelExt,
                   str_model,
                   kwargs_lens,
                   kw_extent=None,
                   fast_caustic = True,
                   savename="test_caustics.png"):
    raise RuntimeError("To re-implement in a better way")
    """
    _coords     = LPClass.data_class
    deltaPix    = aClass.deltaPix
    _frame_size = aClass.pixel_num * deltaPix

    f, ax = plt.subplots(2,figsize=(4,8))
    
    # try to load them:
    filename = aClass.savedir+"/caustics.pkl"
    try:
        with open(filename,"rb") as f:
            results = pickle.load(f)
        print("Loaded "+filename)
        ra_crit_list_1stM, dec_crit_list_1stM, ra_caustic_list_1stM, dec_caustic_list_1stM = results
    except FileNotFoundError:    
        print("File not found: "+filename)        
        if fast_caustic:
            ra_crit_list, dec_crit_list_, ra_caustic_list_1stM, dec_caustic_list_1stM = lensModelExt.critical_curve_caustics(kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix)
        else:
            raise RuntimeError("Doesn't output caustics")

            ra_crit_list, dec_crit_list = lensModelExt.critical_curve_tiling(kwargs_lens, compute_window=_frame_size,
                                                                         start_scale=deltaPix, max_order=10)
        results = ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list
        with open(filename,"wb") as f:
            pickle.dump(results,f)
        print("Saved "+filename)
    
    plot_util.plot_line_set(ax[0], _coords, ra_caustic_list, dec_caustic_list, color='g')
    ax[0].set_title("Caustic "+str_model)
    plot_util.plot_line_set(ax[1], _coords, ra_crit_list, dec_crit_list, color='r')
    ax[1].set_title("CL "+str_model)
    """
    if kw_extent is None:
        kw_extent = get_extents(LPClass.imageModel,LPClass.arcXkpc) 
    xmin,xmax,ymin,ymax = kw_extent["extent_arcsec"]
    kw_crit = LPClass.critical_curve_caustics()
    cl_rad_x,cl_rad_y   = kw_crit["critical_lines"]["radial"]
    cc_rad_x,cc_rad_y   = kw_crit["caustics"]["radial"]
    cl_tan_x,cl_tan_y   = kw_crit["critical_lines"]["tangential"]
    cc_tan_x,cc_tan_y   = kw_crit["caustics"]["tangential"]

    cent_caust_tan = np.mean(cc_tan_x),np.mean(cc_tan_y)
    fig,ax = plt.subplots()
    ax.scatter(cc_rad_x,cc_rad_y,c="k",marker=".",label="Radial Caustics")
    ax.scatter(cc_tan_x,cc_tan_y,c="y",marker=".",label="Tangential Caustics")
    ax.scatter(*cent_caust_tan_PART,c="k",marker="x",label="Tang. Center Caustic")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.legend()
    ax.set_title("Caustics Lines") 
    
    plt.tight_layout()
    print("Saving "+savename) 
    plt.savefig(savename)
    plt.close()
    
def plot_caustics(Model,fast_caustic = True,savename="test_caustics.png",skip_show=False):
    lensModelExt = LensModelExtensions(Model.lens_model_PART)
    kwargs_lens  = Model.kwargs_lens_PART

    str_model    = "PM"

    return _plot_caustics(Model,
                          lensModelExt,str_model,kwargs_lens,
                          fast_caustic=fast_caustic,savename=savename,skip_show=skip_show)

"""
def get_kappa(Model,plot=True,savename="comp_kappa.png",skip_show=False):
    if "tcAS" in Model.PMLens.name:
        raise RuntimeError("this is not the real kappa, but the histogram of the particle as if they had no size")

    kw_samples = Gal2kw_samples(Model.Gal,Model.radius,Model.proj_index,Model.arcXkpc)
    samples    = kw_samples["RAs"],kw_samples["DECs"]
    Ms         = kw_samples["Ms"]
    return _get_kappa(imageModel = Model.imageModel,
                      samples    = samples,
                      Ms         = Ms,
                      arcXkpc    = Model.arcXkpc,
                      SigCrit    = Model.SigCrit,
                      plot       = plot,
                      savename   = savename,
                      skip_show  = skip_show)
"""
def get_extents(Model,arcXkpc):
    _ra,_dec = Model.imageModel.ImageNumerics.coordinates_evaluate #arcsecs 
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
    kw_ext = {"extent_kpc":extent_kpc,
              "extent_arcsec":extent_arcsec,
              "bins_arcsec":bins_arcsec,
              "DRaDec":[Dra01,Ddec01]}
    return kw_ext
"""
def _get_kappa(imageModel,samples,Ms,
                          arcXkpc,SigCrit,
                          plot=True,savename="kappa.png",skip_show=True):
    kw_ext = get_extents(imageModel,arcXkpc)

    RAs,DECs                    = samples
    
    mass_grid, xedges, yedges   = np.histogram2d(RAs,DECs,
                                       bins=kw_ext["bins_arcsec"],
                                       weights=Ms,
                                       density=False) 
    # mass_grid shape: (nx, ny) -> transpose to (ny, nx) -> given the circular simmetry, doesn't really matter

    Dra01,Ddec01 = kw_ext["DRaDec"]
    # density_ij = M_ij/(Area_bin_ij)
    density    = mass_grid.T / (Dra01*Ddec01/(arcXkpc**2)) # Msun/kpc^2
    
    kappa_PART = density/SigCrit
    kappa_PART = kappa_PART.to("").value
    
    if plot:
        plot_kappamap(kappa_PART,extent_kpc=kw_ext["extent_kpc"],title1=r"$\kappa$ Part.",savename=savename,skip_show=skip_show)
    return kappa_PART,kw_ext

"""
def plot_kappamap(kappa1,extent_kpc,title1="",savename="kappa.png",skip_show=False):
    fig,axes = plt.subplots(2,figsize=(8,16))

    ax  = axes[0]
    im0 = ax.imshow(kappa1,origin="lower",extent=extent_kpc)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(title1) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')


    # take advantage of the circular simmetry and obtain the projection
    k1_proj = kappa1[int(len(kappa1)/2)]
    x = np.linspace(extent_kpc[0],extent_kpc[1],len(k1_proj))
    _xcnt = np.median(x)
    ax = axes[1]
    ax.plot(x,k1_proj,c="k")
    fwhm_k1 = get_fwhm(k1_proj,x) 
    hmax  = max(k1_proj)/2.
    ax.axvline(_xcnt,c="g",alpha=.5)

    ax.plot([_xcnt-fwhm_k1/2,_xcnt+fwhm_k1/2],[hmax,hmax],ls="-.",c="r",label="FWHM="+str(np.round(fwhm_k1,3)))
    ax.legend()
    ax.set_title(title1 +" projection at x=0")
    plt.suptitle("Density distribution")
    print("Saving "+savename)
    plt.savefig(savename)
    if not skip_show:
        plt.show()
    plt.close()


def plot_lensed_im_and_kappa(Model,savename="lensed_im.pdf",kw_extents=None):
    #kappa,kw_extents = get_kappa(Model,plot=False)
    kappa = Model.kappa_map
    if kw_extents is None:
        kw_extents = get_extents(Model,Model.arcXkpc)
    fg,axes = plt.subplots(1,2,figsize=(10,5))
    ax = axes[0]

    extent_kpc    = kw_extents["extent_kpc"]
    extent_arcsec = kw_extents["extent_arcsec"]
    
    im0   = ax.matshow(kappa,origin='lower',extent=extent_kpc,cmap="hot")
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(r"Convergence "+Model.Gal.Name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"$\kappa$")

    lnsd_im  = Model.image_sim 
    ax = axes[1]
    im0   = ax.matshow(np.log10(lnsd_im)*10,origin='lower',extent=extent_arcsec)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    
    ax.set_title("Lensed image "+Model.Gal.Name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"log$_{10}$ flux [arbitrary]")
    plt.suptitle(r"With z$_{\text{lens}}$="+str(np.round(Model.z_lens,2))+" z$_{\text{source}}$="+str(np.round(Model.z_source,2)))
    plt.tight_layout()
    print("Saving "+savename) 
    plt.savefig(savename)
    plt.close()
    
def plot_all(Model,savename_lensed="lensed_im.pdf",savename_kappa="kappa.png",savename_caustics="caustics.png",fast_caustic=True,skip_caustic=False):
    
    #plot_lensed_im(Model,savename=Model.savedir+"/"+savename_lensed,skip_show=skip_show)
    #get_kappa(Model,savename=Model.savedir+"/"+savename_kappa,skip_show=skip_show)
    kw_extents = get_extents(Model,Model.arcXkpc)
    plot_lensed_im_and_kappa(Model,savename=Model.savedir+"/"+savename_lensed,kw_extents=kw_extents)
    if not skip_caustic:
        plot_caustics(Model,fast_caustic=fast_caustic,savename=Model.savedir+"/"+savename_caustics,kw_extents=kw_extents)
    plt.close()
    return 0
# for compatibility reason with previous versions:
Lens_PART = LensPart



if __name__ == "__main__":
    print("Do not run this script, but test_Gen_PM_PLL_alpha.py")
    exit()