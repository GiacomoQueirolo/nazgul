# copy from PM_lensmodel_0.1.py

import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from python_tools.tools import mkdir
from fnct import std_sim

from ParticleGalaxy import get_rnd_PG,get_lens_dir,get_z_source,get_dP 
from PG_proj_part_hist import prep_Gal,get_dens_map_rotate_hist

Gal = get_rnd_PG()
z_lens = Gal.z
default_cosmo   =  Gal.cosmo#FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
Gal = prep_Gal(Gal)
# if this
dens_Ms_kpc2,radius,dP,dxdy,z_source,cosmo,proj_index = get_dens_map_rotate_hist(Gal,plot=False)


lens_dir = get_lens_dir(Gal)
#TODO: create link to original Gal
kw_lns = lens_dir+"/sim_kwlens.pkl"
kw_lensmodel_data = lens_dir+"/kwdata.pkl"
# cosmo from https://academic.oup.com/mnras/article/474/3/3391/4644836, Agnello 2017

# point mass theta_E (from eq.4.7 of Meneghetti's lecture note - and by memory)
# theta_E = \sqrt ( 4GM D_ls / (c^2 Ds Dl) )
# split the computation such that it's done only once
def thetaE_PM_prefact(z_lens,z_source,cosmo=default_cosmo):
    # from eq 21 of Narayan "Lectures on GL" 2008
    # theta_E(PM) = sqrt(4GM Dds/(c^2 Dd Ds))   = sqrt(M) * sqrt(4G Dds/(c^2 Dd Ds))
    cosmo_ds  = cosmo.angular_diameter_distance(z_source)
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens)
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source)
    pref      = 4*const.G*cosmo_dds/(const.c*const.c*cosmo_ds*cosmo_dd)
    return np.sqrt(pref.to("1/g")) # 

@u.quantity_input
def thetaE_PM(M:u.g,theta_pref:u.g**-.5):
    thetaE_rad = np.sqrt(M)*theta_pref
    thetaE     = thetaE_rad.to("")*u.rad.to("arcsec")
    return thetaE.value #in arcsec

#while True:

print("Selected Gal:",Gal)
thetaE_pref = thetaE_PM_prefact(z_lens=Gal.z,z_source=z_source)
#thetaE_pref = thetaE_pref.to("1/Msun(1/2)").value #convert in 1/sqrt(Msun)

Mstar = Gal.stars["mass"] #should already be in Msun 
Mgas = Gal.gas["mass"]    #should already be in Msun 
Mdm = Gal.dm["mass"]      #should already be in Msun 
Mbh = Gal.bh["mass"]      #should already be in Msun 
Ms = np.concatenate([Mstar,Mgas,Mdm,Mbh])*u.Msun #Msun
"""
try:
    thetaEs = thetaE_PM(Ms,thetaE_pref)
except TypeError:
    Ms      = Ms*const.M_sun
    thetaEs = thetaE_PM(Ms,thetaE_pref)
"""
thetaEs = thetaE_PM(Ms,thetaE_pref)
# project along z axis 
# centered around 0
# convert in arcsec

arcXkpc = default_cosmo.arcsec_per_kpc_proper(Gal.z)


Xstar,Ystar,Zstar =  np.transpose(Gal.stars["coords"])# Mpc
Xgas,Ygas,Zgas    =  np.transpose(Gal.gas["coords"]) # Mpc
Xdm,Ydm,Zdm       =  np.transpose(Gal.dm["coords"]) # Mpc
Xbh,Ybh,Zbh       =  np.transpose(Gal.bh["coords"]) # Mpc
Xs = np.concatenate([Xstar,Xgas,Xdm,Xbh])*u.Mpc.to("kpc")
Ys = np.concatenate([Ystar,Ygas,Ydm,Ybh])*u.Mpc.to("kpc")
Zs = np.concatenate([Zstar,Zgas,Zdm,Zbh])*u.Mpc.to("kpc")

# projection along given indexes
# xy : ind=0
# xz : ind=1
# yz : ind=2
if proj_index==0:
    _=True # all as usual
elif proj_index==1:
    Ys = copy(Zs)
elif proj_index==2:
    Xs  = copy(Ys)
    Ys  = copy(Zs)
    
RAs  = Xs*arcXkpc.to('arcsec/kpc')
DECs = Ys*arcXkpc.to('arcsec/kpc')
RAs  = RAs.value
DECs = DECs.value
RA_cm  = np.sum(RAs* Ms.value)/np.sum(Ms.value)
DEC_cm = np.sum(DECs* Ms.value)/np.sum(Ms.value)


#lens_model_list = []
lens_model_list  = ["POINT_MASS"]*len(thetaEs)

#use CGPT parallelisation:
from concurrent.futures import ThreadPoolExecutor

def build_kwargs_lens(args):
    tE, ra, dec = args
    return {
        "theta_E": tE,
        "center_x": ra,
        "center_y": dec
    }

# Parallel execution
with ThreadPoolExecutor() as executor:
    kwargs_lens = list(executor.map(build_kwargs_lens, zip(thetaEs, RAs, DECs)))

# save the kwargs
print("Saving "+kw_lns)
with open(kw_lns,"wb") as f:
    pickle.dump(kwargs_lens,f)

## Simulate the lens image

import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

lens_model_class = LensModel(lens_model_list=lens_model_list)

# data specifics
background_rms = 0 #.5  # background noise per pixel
exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)

# Define a "sweet spot" for the numpix and the DeltaPix given the app. size of the lens
#Diam_arcsec = np.mean([np.max(RAs)-np.min(RAs),np.max(DECs)-np.min(DECs)])
#print("Size of the lens in arcsecs",Diam_arcsec)
rad = 70*u.kpc 
print(f"WRN: Consider an arbitrary radius of {rad} around the centre")
rad_arcsec  = rad*arcXkpc.to('arcsec/kpc')
Diam_arcsec = 2*rad_arcsec.value #diameter in arcsec

deltaPix = 0.04  # pixel size in arcsec (area per pixel = deltaPix**2)
numPix   = int(.7*Diam_arcsec/deltaPix)
if numPix >500:
    print("Numpix too high: ",numPix,", capping at 500")
    numPix = 500
    print("deltaPix:",deltaPix)
else:
    print("Resulting image size:", numPix)

kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, 
                                             exp_time,background_rms,
                                             center_ra=RA_cm,center_dec=DEC_cm)
data_class = ImageData(**kwargs_data)
kwargs_psf = {'psf_type': 'NONE'} # no PSF for now
        #, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}
psf_class = PSF(**kwargs_psf)

# Source Params
source_model_list = ['SERSIC_ELLIPSE']
ra_source, dec_source = RA_cm,DEC_cm
print("RA,DEC CMS:",RA_cm,DEC_cm)
kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 
                         'center_x': ra_source,
                         'center_y': dec_source, 
                         'e1': -0.1, 'e2': 0.01}

kwargs_source      = [kwargs_sersic_ellipse]
source_model_class = LightModel(light_model_list=source_model_list)
kwargs_numerics    = {'supersampling_factor': 1, 'supersampling_convolution': False}

LensedimageModel = ImageModel(data_class, psf_class, lens_model_class,
                              source_model_class,lens_light_model_class=None,
                              point_source_class=None, kwargs_numerics=kwargs_numerics)
lensed_image_sim = LensedimageModel.image(kwargs_lens, kwargs_source, 
                                          kwargs_lens_light=None, kwargs_ps=None)
SourceimageModel = ImageModel(data_class, psf_class, lens_model_class=None, source_model_class=source_model_class,
                        lens_light_model_class=None,
                        point_source_class=None, kwargs_numerics=kwargs_numerics)
unlensed_image_sim = SourceimageModel.image(kwargs_lens=None, 
                                            kwargs_source=kwargs_source, 
                                            kwargs_lens_light=None,
                                            kwargs_ps=None)

extent = [-rad_arcsec.value,rad_arcsec.value,-rad_arcsec.value,rad_arcsec.value]

fg,ax=plt.subplots(1,2,figsize=(16,8))

ax[0].matshow(np.log10(unlensed_image_sim),extent=extent, origin='lower')
ax[0].contour(np.log10(unlensed_image_sim),cmap=plt.cm.inferno,extent=extent)

ax[0].set_title("Un-Lensed image (lnstr)")

ax[1].matshow(np.log10(lensed_image_sim),extent=extent, origin='lower')
ax[1].contour(np.log10(lensed_image_sim),cmap=plt.cm.inferno,extent=extent)
ax[1].set_title("Lensed image (lnstr)")

name_file = lens_dir+"/lensed_im_PM.pdf"
plt.savefig(name_file)
print("Saving "+name_file)
name_file = "./tmp/lensed_im_PM.pdf"
plt.savefig(name_file)
print("Saving "+name_file)


from lenstronomy.Util import util

x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)  
x,y = util.array2image(x_grid),util.array2image(y_grid)

# deflections
_alpha_x,_alpha_y = lens_model_class.alpha(x_grid, y_grid, kwargs_lens)  
alpha_x,alpha_y   = util.array2image(_alpha_x),util.array2image(_alpha_y)

fg,ax=plt.subplots(1,2,figsize=(16,8))
ax[0].imshow(np.log10(alpha_x),extent=extent,origin="lower")
ax[0].set_title(r"$\alpha_x$")
ax[1].imshow(np.log10(alpha_y),extent=extent,origin="lower")
#ax[0].contour(np.log10(alpha_x),cmap=plt.cm.inferno,extent=extent)
ax[1].set_title(r"$\alpha_y$")
name_file = "./tmp/alpha_PM.pdf"
plt.savefig(name_file)
print("Saving "+name_file)

print("Saving "+kw_lensmodel_data)
with open(kw_lensmodel_data,"wb") as f:
    pickle.dump(kwargs_data,f)