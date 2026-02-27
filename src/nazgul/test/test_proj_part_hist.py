# test of proj_part_hist
# to see what the issue is with the y-coordinates

import os
import csv
import glob
import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
from scipy.stats import gaussian_kde

from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel import convergence_integrals

#from lenstronomy.Util import util

from fnct import gal_dir,std_sim,test_sim
from get_gal_indexes import get_rnd_gal

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


z_source_max = 5
verbose      = True
pixel_num    = 100j
################################################
# debugging funct.

from proj_part import get_radius,get_z_source,get_dP

def get_dens_map_rotate_hist(Gal,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,plot=True):
    # try all projection in order to obtain a lens
    proj_index = 0
    res = None
    res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose,plot=plot)
    raise RuntimeError("DEBUG--Arrived here")
    """
    while proj_index<3:
        try:
            res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose)
            break
        except AttributeError as Ae:
            print("Error : ")
            print(Ae)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if res is None:
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        return res
    """
        
def get_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)

    # given a projection, produce the density map
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
    
    Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc/h
    Xgas,Ygas,Zgas    = Gal.gas["coords"].T   # in Mpc/h
    Xdm,Ydm,Zdm       = Gal.dm["coords"].T    # in Mpc/h
    Xbh,Ybh,Zbh       = Gal.bh["coords"].T    # in Mpc/h
    
    Mstar = Gal.stars["mass"] # in Msun 
    Mgas  = Gal.gas["mass"]  # in Msun 
    Mdm   = Gal.dm["mass"] # in Msun 
    Mbh   = Gal.bh["mass"] # in Msun 
    
    # center around the center of the galaxy
    # correct from cMpc/h to Mpc/h
    # then from Mpc/h to Mpc
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 -> given that it is not corrected, it should prob
    # also be corrected for h -> or maybe not?
    Cx,Cy,Cz= Gal.centre*u.Mpc/(Gal.xy_propr2comov*Gal.h) # this should be now in Mpc
    

    print("DEBUG")
    fig, ax = plt.subplots(3)
    for XX,YY,MM,name in zip([Xstar,Xgas,Xdm,Xbh],[Ystar,Ygas,Ydm,Ybh],[Mstar,Mgas,Mdm,Mbh],["star","gas","dm","bh"]):
        x,y = XX*u.Mpc-Cx,YY*u.Mpc-Cy
        """radius = get_radius(x,y)
        xmin = -radius
        ymin = -radius
        xmax = +radius
        ymax = +radius
        """
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[0].set_xlabel("X [kpc]")
        #ax[0].set_xlim([xmin,xmax])
        ax[1].hist(y,bins=ny,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[1].set_xlabel("Y [kpc]")
        #ax[1].set_xlim([ymin,ymax])
        ax[2].hist(MM/1e8,alpha=.5,label=name)
        ax[2].set_xlabel("M [1e8 SolMass]")
        ax[2].legend()
    namefig = f"{Gal.proj_dir}/hist1D_{proj_index}_part.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig) 



from fnct import Galaxy

if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default="proj_part_hist")
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    parser.add_argument("-zsm","--z_source_max",dest="z_source_max",type=float, help="Maximum source redshift",default=z_source_max)
    parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
                        default=True,action="store_false",help="if True, rerun code")

    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = args.verbose
    z_source_max  = args.z_source_max

        #print("DEBUG -- USING test sym")
        #Gal = get_rnd_gal(sim=test_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
    Gal = get_rnd_gal(sim=std_sim,check_prev=True,reuse_previous=False,min_mass="1e13",max_z="1")
    #gal = load_whatever("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_22/Gn1SGn0.pkl")
    #gal = load_whatever("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_24/Gn1SGn0.pkl")
    #Gal  = Galaxy(2,0,snap=28,CMx=5.21395,CMy=14.088762,CMz=11.199067,M=10392054000000.0)
    Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
    mkdir(Gal.proj_dir)
    Gal.dens_res = f"{Gal.proj_dir}/dens_res.pkl"
    
    dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=Gal,pixel_num=pixel_num,
                                                           z_source_max=z_source_max,verbose=True)#verbose=verbose)

    print("Success")
