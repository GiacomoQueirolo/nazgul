"""
Define the basic galaxy class PartGal (storing all particle data), as well as related helper functions, e.g. to sample galaxies 
-> generalised for COLIBRE and EAGLE simulations
"""
import dill
import numpy as np
from decimal import Decimal
from astropy.stats import sigma_clip

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass

from nazgul.basic_gal import BasicGal,store_class
from nazgul.pathfinder import path_nazgul

class BasicPartGal(BasicGal):
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units
    """
    _large_attributes_setup = []
    _large_attributes_unpack = []
    
    @property
    def name(self):
        """Define name of instance
        """
        raise NotImplementedError
        
    @property
    def dill_path(self):
        """Define dill path to store the class instance
        """
        dill_path = self.gal_dir/f"{self.name}.dll"
        return dill_path
        
    ### Class Structure ####
    ########################
    def __str__(self):
        str_gal = f"Sim {self.sim}"
        str_gal = f"Gal {self.name}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(int(self.N_part))} part.\nof \ntot Mass={'%.1E'%Decimal(float(self.M))} [M_sun]\n"
        return str_gal 
    ########################
    ########################
    
    @property 
    def cosmo(self):
        raise NotImplementedError
        
    def run(self,reload=True):
        raise NotImplementedError
            
    def initialise_parts(self):
        raise NotImplementedError
        
    def _verify_cnt(self):
        """verify that the center of mass is indeed correct
        """
        raise NotImplementedError

    def store_gal(self):
        raise NotImplementedError

def clip_coord(m,x,y,z,sigma=10):
    # clip coordinates outliers
    dists = np.sum(np.array([x,y,z])**2,axis=0)
    mask = np.invert(sigma_clip(dists,sigma=sigma).mask)
    #mask = np.ones(len(x),dtype=bool)
    #for coord in x,y,z:
    #    mask *= np.invert(sigma_clip(coord,sigma=sigma).mask)
    perc_final = np.round(len(m[mask])*100/len(m),3)
    if perc_final<99:
        print(100-perc_final,"% of particle discarded")
        raise RuntimeError("Too many particles discarded")
        
    return m[mask],x[mask],y[mask],z[mask]
    

