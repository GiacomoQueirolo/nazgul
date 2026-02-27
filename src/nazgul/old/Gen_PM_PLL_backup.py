# copy from General_funct_parall.py -> delete kwargs_lens_PART before saving and re-compute it from samples
# when loaded - > need to change how get_kwrg_prm works or find another way to do it

import os,pickle
from scipy.stats import norm
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt
#from lenstronomy.LensModel.lens_model import LensModel 
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


from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Plots import plot_util

from python_tools.fwhm import get_fwhm
from python_tools.tools import mkdir,short_SciNot,to_dimless



pixel_num   = 200 # pix



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

from remade_gal import get_rnd_NG,get_lens_dir,prep_Gal

#
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

def get_LensSystem_kwrgs(deltaPix,pixel_num=pixel_num,background_rms=0.005,exp_time=500):
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
    source_model_list = ['SERSIC_ELLIPSE']
    ra_source, dec_source = 0., 0.
    kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 
                            'center_x': ra_source,
                             'center_y': dec_source, 
                             'e1': -0.1, 'e2': 0.01}

    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # for modelling later:
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 
                     'multi_band_type': 'single-band'}
    return data_class, psf_class, source_model_class, kwargs_numerics, kwargs_source,kwargs_data_joint


##########################
# Model class for parts. #
##########################
from NG_proj_part_hist import get_dens_map_rotate_hist
theta_c_AS     = 5e-3 
kwlens_part_AS = {"type":"AS","theta_cAS":theta_c_AS}
kwlens_part_PM = {"type":"PM"}
from time import time # for DEBUG
from remade_gal import get_rough_radius
from project_gal import #  prep_Gal_denspath

class Lens_PART(): 
    def __init__(self,
                    Galaxy,
                    kwlens_part, # if PM or AS, and if so size of the core
                    radius=2, # probably could compute it in a smart way... -> prob. read it from Galaxy-> Still to well define -> def in arcsec->def 3''
                    pixel_num=pixel_num, # sim prms 
                    #z_lens=z_lens,z_source=z_source,cosmo=cosmo, # cosmo prms -> obtained from Galaxy 
                    exp_time=500,bckg_rms=0.01, # observation parameters
                    savedir_sim="lensing",reload=True # saving params
                    ):
        #Galaxy           = prep_Gal_denspath(Galaxy) # just set up some directories -> maybe not needed
        
        self.Gal         = Galaxy
        lens_dir         = get_lens_dir(self.Gal)
        self.savedir     = f"{lens_dir}/{savedir_sim}"

        # First, obtain the z_source and projection index
        # cosmo prms
        self.cosmo       = self.Gal.cosmo
        self.z_lens      = self.Gal.z
        # Obtain "a priori" the z_source and 1 projection index s.t. Gal is a lens
        self.kwres_dens  = get_dens_map_rotate_hist(self.Gal,plot=True)
        self.z_source    = self.kwres_dens["z_source"]
        self.proj_index  = self.kwres_dens["proj_index"]

        self.arcXkpc     = self.cosmo.arcsec_per_kpc_proper(self.z_lens)
        self.SigCrit     = SigCrit(cosmo=self.cosmo,z_lens=self.z_lens,z_source=self.z_source) # Msun/kpc^2

        # Then define the radius based on ~ theta_E
        t0 = time()
        self.radius      = get_rough_radius(self.Gal,self.proj_index,self.z_source,scale=2) #in arcsec
        t1 = time()
        print("DEBUG:\nradius=",self.radius)
        print("Computation time for rough radius",t1-t0)
        self.pixel_num   = pixel_num        
        Diam_arcsec      = 2*self.radius #diameter in arcsec
        self.deltaPix    = Diam_arcsec/self.pixel_num # ''/pix

        # kwargs_lens for the particles :
        # type:"AS" or "PM"
        # if AS: param:{"thetacAS"}
        self.PMLens      = PMLens(kwlens_part)
        self.PMLens.setup(self)

        # observational params
        self.exp_time  =  exp_time
        self.bckg_rms  =  bckg_rms
        
        mkdir(self.savedir)
        self._setup_names()
        self.reload    = reload
        
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
        prev_mod = ReadClass(self)
        if prev_mod is False:
            return False
        else:
            # verify that the inputs are the same:
            for k in self.__dict__.keys():
                if prev_mod.__dict__[k]!=self.__dict__[k]:
                    print("But the previous was not exactly the same, rerunning")
                    return False
            for k in prev_mod.__dict__.keys():
                self.__dict__[k] = prev_mod.__dict__[k]
            return True
        
    def run(self,read_prev=True):
        upload_successful = False
        if read_prev:
            upload_successful = self.upload_prev()
        if not upload_successful:
            self.setup_lenses()
            # images:
            self.setup_dataclasses()
            self.imageModel_PART,self.image_sim_PART  = self.get_lensed_image()
            self.store()

    def setup_dataclasses(self):
        self.data_class,self.psf_class,self.source_model_class,\
        self.kwargs_numerics, self.kwargs_source,self.kwargs_data_joint = \
                get_LensSystem_kwrgs(self.deltaPix,self.pixel_num,background_rms=self.bckg_rms,exp_time=self.exp_time)
        return 0

    # the following is meant to be rerun every time we load the class to save space
    # -> computationally not too intense (medium tho)
    def setup_lenses(self):
        # Convert x,y,z in samples and get masses
        kw_samples = Gal2kw_samples(self.Gal,self.radius,self.proj_index,self.arcXkpc)
        samples    = kw_samples["RAs"],kw_samples["DECs"]
        Ms         = kw_samples["Ms"]
        # Convert in lenses parameters
        kwLns_PART,LnsMod_PART = self.PMLens.get_lens_PART(samples=samples,Ms=Ms)
        self.kwargs_lens_PART  = kwLns_PART
        self.lens_model_PART   = LnsMod_PART
        return 0
        
    def get_lensed_image(self):
        imageModel = ImageModel(self.data_class, self.psf_class, 
                                self.lens_model_PART, 
                                self.source_model_class,lens_light_model_class=None,point_source_class=None, 
                                kwargs_numerics=self.kwargs_numerics)
        image_sim  = imageModel.image(self.kwargs_lens_PART, self.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
        return imageModel,image_sim

    def get_alpha_map(self):
        RA,DEC = self.get_RADEC()
        alpha_x,alpha_y = self.lens_model_PART.alpha(RA, DEC, self.kwargs_lens_PART)
        # once computed, should be stored
        self.alpha_x,self.alpha_y = alpha_x,alpha_y 
        return alpha_x,alpha_y
    
    def get_RADEC(self):
        _ra,_dec = self.imageModel_PART.ImageNumerics.coordinates_evaluate #arcsecs  
        RA,DEC   = util.array2image(_ra),util.array2image(_dec)
        return RA,DEC
        
    def update_kwargs_data_joint(self,add_noise=False):
        # updating it with the PM image
        if add_noise:
            image   = self.image_sim_PART
            poisson = image_util.add_poisson(image, exp_time=self.exp_time)
            bkg     = image_util.add_background(image, sigma_bkd=self.bckg_rms)
            
            self.image_sim_PART = image + poisson + bkg

            #other realisation of the bkg
            bkg       = image_util.add_background(image, sigma_bkd=self.bckg_rms)
            noise_map =  np.sqrt(poisson**2+bkg**2)
            self.kwargs_data_joint["multi_band_list"][0][0]["noise_map"] = noise_map
        self.kwargs_data_joint["multi_band_list"][0][0]["image_data"] = self.image_sim_PART
            
    def __str__(self):
        if not hasattr(self,"name",False):
            self._setup_names()
        return self.name
        
    def store(self):
        if not hasattr(self,"pkl_path"):
            self._setup_names()
        # store this sim
        # but not the most heavy parts
        self_copy = deepcopy(self)
        # delete kwargs_lens_PART 
        # -> can be recovered fast from the Gal with setup_lenses
        del self_copy.kwargs_lens_PART
        # delete imagenumerics heaviest component (fast to recover)
        self_copy.imageModel_PART.ImageNumerics._numerics_subframe._grid = None

        with open(self.pkl_path,"wb") as f:
            pickle.dump(self_copy,f)
        print("Saved "+self.pkl_path)

from remade_gal import part2RaDecM,get_CM

def Gal2kw_samples(Gal,radius,proj_index,arcXkpc,nbins=100):
    Ms,RAs,DECs = part2RaDecM(Gal,proj_index,arcXkpc=arcXkpc)
    # RA,DEC= arcsec, Ms = Msun
    try:
        RAs,DECs = RAs.value,DECs.value 
    except AttributeError:
        pass
    print("Some galaxy have a 'shifted' CM")
    RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    
    kw_samples   = {}
    #print("We recenter around CM)#no it was not done previously
    print("We recenter around ~CM~: no, the densest point within a cercle of radius rad from the CM") 

    print("DEBUG")
    print("RA_cm,radius")
    print(RA_cm,radius)
    bins = [np.linspace(RA_cm - radius,RA_cm+radius,nbins),
            np.linspace(DEC_cm - radius,DEC_cm+radius,nbins)]
    
    mass_grid, xedges, yedges   = np.histogram2d(RAs,DECs,
                                       bins=bins,
                                       weights=Ms,
                                       density=False)
    # max density indexes
    ix, iy = np.unravel_index(np.argmax(mass_grid), mass_grid.shape)
    
    # 2. Compute center coordinates
    RA_dns  = 0.5 * (xedges[ix] + xedges[ix+1])
    DEC_dns = 0.5 * (yedges[iy] + yedges[iy+1])
    print("Info:  CM vs Densest ")
    print("CM:",RA_cm,DEC_cm)
    print("Dns:",RA_dns,DEC_dns)

    kw_samples["RAs"]  = RAs-RA_dns   #arcsec
    kw_samples["DECs"] = DECs-DEC_dns  #arcsec
    
    #kw_samples["RAs"]  = RAs-RA_cm   #arcsec
    #kw_samples["DECs"] = DECs-DEC_cm  #arcsec
    kw_samples["Ms"]   = Ms    #Msun
    kw_samples["cm"]   = RA_cm-RA_dns,DEC_cm-DEC_dns  # 

    
    return kw_samples
    
#
# helper funct
#

# this function is a wrapper for convenience - it takes the class itself as input
def ReadClass(aClass,verbose=True):
    return LoadClass(aClass.pkl_path,verbose=verbose)

def LoadClass(path,verbose=True):
    if os.path.isfile(path):
        print("File "+path+" is present")
        try:
            return _LoadClass(path=path,verbose=verbose)
        except Exception as e:
            print("But failed to load: "+str(e))
            return False
    else:
        print("File not present")
        return False
        
def _LoadClass(path,verbose=True):
    with open(path,"rb") as f:
        aClass = pickle.load(f)
    gridClassType = getattr(aClass.kwargs_numerics,"compute_mode","regular")
    if gridClassType =="regular":
        from lenstronomy.ImSim.Numerics.grid import RegularGrid as Grid
    elif gridClassType  == "adaptive":
        from lenstronomy.ImSim.Numerics.grid import AdaptiveGrid as Grid
    recomputed_grid = Grid(nx=aClass.pixel_num,ny=aClass.pixel_num,
    transform_pix2angle=aClass.imageModel_PART.Data.transform_pix2angle,
    ra_at_xy_0=aClass.imageModel_PART.Data.radec_at_xy_0[0],
    dec_at_xy_0=aClass.imageModel_PART.Data.radec_at_xy_0[1])
    aClass.imageModel_PART.ImageNumerics._numerics_subframe._grid = recomputed_grid 
    # recover also kwargs_lens_PART:
    aClass.setup_lenses() 
    if verbose:
        print(f"Loaded {aClass.pkl_path}")
    return aClass



def _plot_caustics(aClass,
                   lensModelExt_1stModel,
                   str_1stModel,
                   kwargs_lens_1stM,
                   fast_caustic = True,savename="test_caustics.png",skip_show=False):
    _coords     = aClass.data_class
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
            ra_crit_list_1stM, dec_crit_list_1stM, ra_caustic_list_1stM, dec_caustic_list_1stM = lensModelExt_1stModel.critical_curve_caustics(kwargs_lens_1stM, compute_window=_frame_size, grid_scale=deltaPix)
        else:
            raise RuntimeError("Doesn't output caustics")

            ra_crit_list_1stM, dec_crit_list_1stM = lensModelExt_1stModel.critical_curve_tiling(kwargs_lens_1stM, compute_window=_frame_size,
                                                                         start_scale=deltaPix, max_order=10)
        results = ra_crit_list_1stM, dec_crit_list_1stM, ra_caustic_list_1stM, dec_caustic_list_1stM
        with open(filename,"wb") as f:
            pickle.dump(results,f)
        print("Saved "+filename)
            
    plot_util.plot_line_set(ax[0], _coords, ra_caustic_list_1stM, dec_caustic_list_1stM, color='g')
    ax[0].set_title("Caustic "+str_1stModel)
    plot_util.plot_line_set(ax[1], _coords, ra_crit_list_1stM, dec_crit_list_1stM, color='r')
    ax[1].set_title("CL "+str_1stModel)
    print("Saving "+savename) 
    plt.savefig(savename)
    if not skip_show:
        plt.show()
    plt.close()
    
def plot_caustics(Model,fast_caustic = True,savename="test_caustics.png",skip_show=False):
    lensModelExt_1stModel = LensModelExtensions(Model.lens_model_PART)
    kwargs_lens_1stM      = Model.kwargs_lens_PART

    str_1stModel = "PM"

    return _plot_caustics(Model,
                          lensModelExt_1stModel,str_1stModel,kwargs_lens_1stM,
                          fast_caustic=fast_caustic,savename=savename,skip_show=skip_show)


def get_kappa(Model,plot=True,savename="comp_kappa.png",skip_show=False):
    kw_samples = Gal2kw_samples(Model.Gal,Model.radius,Model.proj_index,Model.arcXkpc)
    samples    = kw_samples["RAs"],kw_samples["DECs"]
    Ms         = kw_samples["Ms"]
    return _get_kappa(imageModel = Model.imageModel_PART,
                      samples    = samples,
                      Ms         = Ms,
                      arcXkpc    = Model.arcXkpc,
                      SigCrit    = Model.SigCrit,
                      plot       = plot,
                      savename   = savename,
                      skip_show  = skip_show)

def get_extents(imageModel,arcXkpc):
    _ra,_dec = imageModel.ImageNumerics.coordinates_evaluate #arcsecs 
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
    bins_arcsec   = [ra_edges.value,dec_edges.value]
    kw_ext = {"extent_kpc":extent_kpc,
              "extent_arcsec":extent_arcsec,
              "bins_arcsec":bins_arcsec,
              "DRaDec":[Dra01,Ddec01]}
    return kw_ext
    
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


def plot_lensed_im_and_kappa(Model,savename="lensed_im.pdf",skip_show=False):
    kappa,kw_extents = get_kappa(Model,plot=False)
    
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

    lnsd_im  = Model.image_sim_PART 
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
    if not skip_show:
        plt.show()
    plt.close()
    
def plot_all(Model,savename_lensed="lensed_im.pdf",savename_kappa="kappa.png",savename_caustics="caustics.png",fast_caustic=True,skip_show=False,skip_caustic=False):
    
    #plot_lensed_im(Model,savename=Model.savedir+"/"+savename_lensed,skip_show=skip_show)
    #get_kappa(Model,savename=Model.savedir+"/"+savename_kappa,skip_show=skip_show)
    plot_lensed_im_and_kappa(Model,savename=Model.savedir+"/"+savename_lensed,skip_show=skip_show)
    if not skip_caustic:
        plot_caustics(Model,fast_caustic=fast_caustic,savename=Model.savedir+"/"+savename_caustics,skip_show=skip_show)
    plt.close()
    return 0


if __name__ == "__main__":
    Gal = get_rnd_NG()
    print("Pixel_num:",pixel_num)
    # for testing we reload the lensing data
    mod_LP = Lens_PART(Galaxy=Gal,kwlens_part=kwlens_part_AS,radius=4,pixel_num=pixel_num,reload=False)
    mod_LP.run()
    plot_all(mod_LP,skip_show=True,skip_caustic=True)
    
