# create the particle data for GLAMER
# file: CSV4
# path : EAGLE_prt_data/particles_EAGLE.csv
# structure: 
# 6 columns :
# x,y,z [Mpc or Mpc/h?], Mass [Msun/h] #, part. smooth size [Mpc/h], type of part [int]
# coord should be in physical Mpc unit -> Mpc only right? -> all values are by default given indep of cosmology (i.e. 1/h) as described in the eagle-particle paper, sec. 2.3.8
# and implemented in the fnct.py, read_dataset function (obtained from the same paper)

import numpy as np
from argparse import ArgumentParser 
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from fnct import std_sim
#from get_gal_indexes import get_rnd_gal
from remade_gal import get_rnd_NG

filename = "particles_EAGLE.csv"
path     = "./EAGLE_prt_data/"

if __name__=="__main__":
    parser = ArgumentParser(description="Produce particle file of galaxy in csv")
    parser.add_argument("-fn",dest="filename",type=str, help="Output filename",default=filename)
    parser.add_argument("-pth","--path",dest="path",type=str, help="Output path",default=path)
    args          = parser.parse_args()
    path          = args.path
    filename      = args.filename
    if ".csv"!=filename[-4:]:
        filename +=".csv"
    #Gal = get_rnd_gal(sim=std_sim,check_prev=False,reuse_previous=False)
    Gal = get_rnd_NG(sim=std_sim,check_prev=False)
    Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc/h
    Xgas,Ygas,Zgas    = Gal.gas["coords"].T # in Mpc/h
    Xdm,Ydm,Zdm       = Gal.dm["coords"].T # in Mpc/h
    Xbh,Ybh,Zbh       = Gal.bh["coords"].T # in Mpc/h
    Mstar = Gal.stars["mass"] #should already be in Msun/h
    Mgas  = Gal.gas["mass"] #should already be in Msun/h
    Mdm   = Gal.dm["mass"] #should already be in Msun/h 
    Mbh   = Gal.bh["mass"] #should already be in Msun/h
    """
    # -> doesn't work bc DM part are not smoothed
    SmoothStar = Gal.stars["smooth"] # in Mpc/h
    SmoothGas  = Gal.gas["smooth"] # in Mpc/h
    SmoothDM   = Gal.dm["smooth"] # in Mpc/h
    SmoothBH   = Gal.bh["smooth"] # in Mpc/h
    """
    
    # Concatenate particle properties
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])/Gal.h #in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])/Gal.h #in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])/Gal.h #in Mpc
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])/Gal.h #in Msun
    #s = np.concatenate([SmoothDM, SmoothStar, SmoothGas, SmoothBH])
    """
    # Particle type index: 1=dm, 2=stars, 3=gas, 4=bh
    ptype_dm   = np.full_like(Xdm,   1, dtype=int)
    ptype_star = np.full_like(Xstar, 2, dtype=int)
    ptype_gas  = np.full_like(Xgas,  3, dtype=int)
    ptype_bh   = np.full_like(Xbh,   4, dtype=int)
    ptype      = np.concatenate([ptype_dm, ptype_star, ptype_gas, ptype_bh])
    """
    # Stack into (N,4) array
    data = np.column_stack([x, y, z, m])
    # have to add a line w 4 columns w the names as the first line (divided by ,)
    header = "x,y,z,M"
    # Save to CSV with no header, float fmt
    np.savetxt(path+filename, data,header=header, fmt="%.6e", delimiter=",")
    print("Saved "+path+filename)
    print("from galaxy:\n"+str(Gal))