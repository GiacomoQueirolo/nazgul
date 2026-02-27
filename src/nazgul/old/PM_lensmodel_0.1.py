# try to model a lens with point masses for the particles:
import pickle
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from python_tools.tools import mkdir
from fnct import std_sim
from get_gal_indexes import get_rnd_gal

# Sim source:
z_source = 2.1
default_cosmo= FlatLambdaCDM(H0=70, Om0=0.3)
Gal = get_rnd_gal(sim=std_sim,check_prev=True,reuse_previous=True)
lens_dir = "sim_lens/"+str(Gal.sim)+"/snap"+str(Gal.snap)+"_G"+str(Gal.Gn)+"."+str(Gal.SGn)+"/"
mkdir(lens_dir) 
#TODO: create link to original Gal
kw_lns = lens_dir+"/sim_kwlens.pkl"
kw_lensmodel_data = lens_dir+"/kwdata.pkl"
# cosmo from https://academic.oup.com/mnras/article/474/3/3391/4644836, Agnello 2017

# divide the computation such that it's done only once
def thetaE_PM_prefact(z_lens,z_source,cosmo=default_cosmo):    
    cosmo_ds  = cosmo.angular_diameter_distance(z_source)
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens)
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source)
    pref      = 4*const.G*cosmo_dds/(const.c*const.c*cosmo_ds*cosmo_dd)
    return np.sqrt(pref) # 

@u.quantity_input
def thetaE_PM(M:u.g,theta_pref:u.g**-.5):
    thetaE_rad = np.sqrt(M)*theta_pref
    thetaE_rad = thetaE_rad.to("")*u.rad
    thetaE     = thetaE_rad.to("arcsec")
    return thetaE.value #in arcsec

#while True:

print("Selected Gal:",Gal)
thetaE_pref = thetaE_PM_prefact(z_lens=Gal.z,z_source=z_source)
#thetaE_pref = thetaE_pref.to("1/Msun(1/2)").value #convert in 1/sqrt(Msun)

Mstar = Gal.stars["mass"] #should already be in Msun 
Mgas = Gal.gas["mass"] #should already be in Msun 
Mdm = Gal.dm["mass"] #should already be in Msun 
Mbh = Gal.bh["mass"] #should already be in Msun 
Ms = np.concatenate([Mstar,Mgas,Mdm,Mbh]) #Msun
try:
    thetaEs = thetaE_PM(Ms,thetaE_pref)
except TypeError:
    Ms      = Ms*const.M_sun
    thetaEs = thetaE_PM(Ms,thetaE_pref)
# project along z axis 
# centered around 0
# convert in arcsec
arcXkpc = default_cosmo.arcsec_per_kpc_proper(Gal.z)

Xstar,Ystar,_ =  np.transpose(Gal.stars["coords"] - Gal.centre )# Mpc
#RAstar,DECstar = Xstar.to("kpc").value*arcXkpc,Ystar.to("kpc").value*arcXkpc
Xgas,Ygas,_ =  np.transpose(Gal.gas["coords"] - Gal.centre) # Mpc
#RAgas,DECgas = Xgas.to("kpc").value*arcXkpc,Ygas.to("kpc").value*arcXkpc
Xdm,Ydm,_ =  np.transpose(Gal.dm["coords"] - Gal.centre ) # Mpc
#RAdm,DECdm = Xdm.to("kpc").value*arcXkpc,Ydm.to("kpc").value*arcXkpc
Xbh,Ybh,_ =  np.transpose(Gal.bh["coords"] - Gal.centre) # Mpc
#RAbh,DECbh = Xbh.to("kpc").value*arcXkpc,Ybh.to("kpc").value*arcXkpc


RAs  = np.concatenate([Xstar,Xgas,Xdm,Xbh]).to("kpc").value*arcXkpc
DECs = np.concatenate([Ystar,Ygas,Ydm,Ybh]).to("kpc").value*arcXkpc

#lens_model_list = []
lens_model_list  = ["POINT_MASS"]*len(thetaEs)
"""
kwargs_lens = []
for tE,ra,dec in zip(thetaEs,RAs,DECs):
    kw_prt = {"theta_E":tE,
              "center_x":ra,
              "center_y":dec
             }
    kwargs_lens.append(kw_prt)
"""
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

lens_model_class = LensModel(lens_model_list=lens_model_list)

# data specifics
background_rms = .5  # background noise per pixel
exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)

# Define a "sweet spot" for the numpix and the DeltaPix given the app. size of the lens
Diam_arcsec = np.mean([np.max(RAs)-np.min(RAs),np.max(DECs)-np.min(DECs)])
print("Size of the lens in arcsecs",Diam_arcsec)

#numPix   = 100   # cutout pixel size
deltaPix = 0.08  # pixel size in arcsec (area per pixel = deltaPix**2)
numPix = 1.5*Diam_arcsec/deltaPix
if numPix >500:
    print("Numpix too high: ",numPix,", capping at 500")
    numPix = 500
    print("deltaPix:",deltaPix)
else:
    print("Resulting image size:", numPix)

#fwhm = 0.01  # full width half max of PSF -> very small, almost none


kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
data_class = ImageData(**kwargs_data)
kwargs_psf = {'psf_type': 'NONE'} # no PSF for now
        #, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}
psf_class = PSF(**kwargs_psf)

# Source Params
source_model_list = ['SERSIC_ELLIPSE']
ra_source, dec_source = 0., 0.
kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 'center_x': ra_source,
                         'center_y': dec_source, 
                         'e1': -0.1, 'e2': 0.01}

kwargs_source = [kwargs_sersic_ellipse]
source_model_class = LightModel(light_model_list=source_model_list)
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                        lens_light_model_class=None,
                        point_source_class=None, kwargs_numerics=kwargs_numerics)
image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
plt.matshow(np.log10(image_sim), origin='lower')
plt.savefig(lens_dir+"/lensed_im.pdf")

poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
bkg = image_util.add_background(image_sim, sigma_bkd=background_rms)
image_sim = image_sim + bkg + poisson

kwargs_data['image_data'] = image_sim
data_class.update_data(image_sim)

plt.matshow(np.log10(image_sim), origin='lower')
plt.savefig(lens_dir+"/lensed_im_noisy.pdf")

print("Saving "+kw_lensmodel_data)
with open(kw_lensmodel_data) as f:
    pickle.dump(kwargs_data,f)