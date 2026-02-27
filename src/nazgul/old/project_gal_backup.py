# Take Gal from remade_gal and does projection and similar calculations
# functions usefuls for lensing and later imported by Gen_PM_PLL.py


import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

from python_tools.tools import mkdir


# for now keep this and check if still needed
dir_name     = "proj_part_hist"
def prep_Gal_denspath(Gal,dir_name=dir_name):
    # impractical but easy to set up
    Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
    mkdir(Gal.proj_dir)
    Gal.dens_res = f"{Gal.proj_dir}/dens_res.pkl"
    return Gal


def get_CM(Ms,RAs,DECs):
    RA_cm  = np.sum(RAs* Ms)/np.sum(Ms)
    DEC_cm = np.sum(DECs* Ms)/np.sum(Ms)
    return RA_cm,DEC_cm

def part2RaDecM(Gal,proj_index,arcXkpc=None):
    # Given the galaxy and a projection index, return Masses (in Msun) and
    # XY coords. of particles
    # without arcXkpc : in kpc (value)
    # with arcXkpc: in arcsec  (value)
    
    # Particle masses
    Mstar = Gal.stars["mass"] # Msun
    Mgas  = Gal.gas["mass"]   # Msun
    Mdm   = Gal.dm["mass"]    # Msun
    Mbh   = Gal.bh["mass"]    # Msun
    Ms    = np.concatenate([Mstar,Mgas,Mdm,Mbh])*u.Msun #Msun

    # Particle pos
    Xstar,Ystar,Zstar =  np.transpose(Gal.stars["coords"]) # Mpc
    Xgas,Ygas,Zgas    =  np.transpose(Gal.gas["coords"])   # Mpc
    Xdm,Ydm,Zdm       =  np.transpose(Gal.dm["coords"])    # Mpc
    Xbh,Ybh,Zbh       =  np.transpose(Gal.bh["coords"])    # Mpc
    Xs = np.concatenate([Xstar,Xgas,Xdm,Xbh])*u.Mpc.to("kpc") #kpc
    Ys = np.concatenate([Ystar,Ygas,Ydm,Ybh])*u.Mpc.to("kpc") #kpc
    Zs = np.concatenate([Zstar,Zgas,Zdm,Zbh])*u.Mpc.to("kpc") #kpc

    # projection along given indexes
    # xy : ind=0
    # xz : ind=1
    # yz : ind=2
    if proj_index==0:
        _   = True  # all as usual
    elif proj_index==1:
        Ys  = copy(Zs)
    elif proj_index==2:
        Xs  = copy(Ys)
        Ys  = copy(Zs)
        
    if arcXkpc is None:
        return Ms,Xs,Ys 
    else:
        RAs    = Xs*arcXkpc.to('arcsec/kpc')
        DECs   = Ys*arcXkpc.to('arcsec/kpc')
        
        return Ms,RAs,DECs


def get_rough_radius(Gal,proj_index,z_source,scale=2):
    # -> this should only be used for plotting
    # the idea is simple:
    # we want a very approximate idea of the theta_E of the galaxy
    # to do that, we fit a SIS to its particle distribution 
    # basically in 1D, assuming (wrong but we don't care) spherical symmetry
    # then we scale that by the scale (default=2) and that is our aperture

    cosmo   = Gal.cosmo
    Dd      = cosmo.angular_diameter_distance(Gal.z).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(Gal.z,z_source).to("Mpc") 
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc")
    
    Ms,RAs,DECs  = part2RaDecM(Gal,proj_index,arcXkpc=arcXkpc)
    RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    # note: RA/DEC are given in arcsec, and Ms in Msun
    RA_centered,DEC_centered = RAs-RA_cm,DECs-DEC_cm
    X_centered,Y_centered = RA_centered/arcXkpc,DEC_centered/arcXkpc
    return scale*theta_E_from_particles(Ms,X_centered,Y_centered,Dd,Ds,Dds)

def _get_rough_radius(Ms,RAs,DECs,Dd,Ds,Dds,scale=2,RA_cm=None,DEC_cm=None):
    if RA_cm is None or DEC_cm is None:
        RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    RA_centered,DEC_centered = RAs-RA_cm,DECs-DEC_cm
    X_centered,Y_centered = RA_centered/arcXkpc,DEC_centered/arcXkpc
    return scale*theta_E_from_particles(Ms,X_centered,Y_centered,Dd,Ds,Dds)


def theta_E_from_particles(Ms, RA, DEC, Dd, Ds, Dds, nbins=200):
    """
    Ms [Msun], RA, DEC [arcsec]
    Returns theta_E in arcsec.
    """
    # big brain time: 
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc")
    # critical surface density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # cylindrical radius
    thetas = np.sqrt(RA**2 + DEC**2)  # arcsec
    
    # mass in annuli
    hist, edges = np.histogram(thetas, bins=nbins, weights=Ms)
    theta_mid = 0.5 * (edges[1:] + edges[:-1])
    
    # enclosed mass
    M_encl = np.cumsum(hist)
    
    # area of circle
    area = np.pi * theta_mid**2 #arcsec^2
    # average Sigma(<R)
    Sigma_encl = M_encl / area # Msun/arcsec^2
    Sigma_encl_kpc2 /= arcXkpc**2  # Msun/kpc^2
    # find R where Sigma_encl = Sigma_crit

    idx = np.argmin(np.abs(Sigma_encl_kpc2 - Sigma_crit))
    theta_E_arcsec = theta_mid[idx] # arcsec
    
    return theta_E_arcsec
"""    
def theta_E_from_particles(Ms, X, Y, Dd, Ds, Dds, nbins=200):
    #Ms [Msun], X,Y [kpc], distances in Mpc.
    #Returns theta_E in arcsec.
    # critical surface density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # cylindrical radius
    R = np.sqrt(X**2 + Y**2)  # kpc
    
    # mass in annuli
    hist, edges = np.histogram(R, bins=nbins, weights=Ms)
    Rmid = 0.5 * (edges[1:] + edges[:-1])
    
    # enclosed mass
    M_encl = np.cumsum(hist)
    
    # area of circle
    area = np.pi * Rmid**2 #kpc
    # average Sigma(<R)
    Sigma_encl = M_encl / area

    # find R where Sigma_encl = Sigma_crit

    idx = np.argmin(np.abs(Sigma_encl - Sigma_crit))
    R_E_kpc = Rmid[idx]

    # convert to angular Einstein radius
    theta_E_rad    = (R_E_kpc * u.kpc / (Dd * u.Mpc)).value*u.rad
    theta_E_arcsec = theta_E_rad.to("arcsec")

    return theta_E_arcsec

# utter shit:
def get_tE_SIS(Ms,X_centered,Y_centered,DsDds,nbins=100):
    # Ms in unit of Msun
    # X_centered,Y_centered in kpc and centered around the density peak
    # DsDds = D_s / D_ds (non-dimensional)
    r_kpc = np.sqrt(X_centered**2 + Y_centered**2) #kpc
    m_bins,bin_edge = np.hist(r_kpc,bins=nbins,weights=Ms)  # [m_bins] = Msun
    area_bins = np.pi*(edges[1:]**2 - edges[:-1]**2)
    Sigma = m_bins/area_bins
    # [Sigma] = Msun/kpc^2
    r_i = (np.diff(bin_edge)*0.5)+bin_edge[:-1] # kpc
    # rho(r) = theta_E*Ds/Dds*(c^2/8pi^2G)*(1/r^2) ->
    # Sigma(r) =  theta_E*Ds/Dds*(c^2/8piG)*(1/r)
    # now, we can take x=(1/r) and fit Sigma as a linear function
    inv_ri       = 1/(r_i) # r_i in kpc -> 1/kpc
    fit_rho,_    = np.polyfit(inv_ri,Sigma,1) # [fit_rho] = Msun/kpc
    pref_theta_E = (DsDds)*(const.c**2)/(8*np.pi*const.G)
    theta_E      = fit_rho/pref_theta_E.to("Msun/kpc").value # radians
    theta_E      = theta_E*u.rad.to("arcsec").value
    return theta_E
"""
