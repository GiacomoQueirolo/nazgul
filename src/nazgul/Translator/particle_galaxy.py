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

class BasicPartGal:
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units
    """
    _large_attributes = []
    @property
    def Name(self):
        """Define name of instance
        """
        raise NotImplementedError
        
    @property
    def dill_path(self):
        """Define dill path to store the class instance
        """
        dill_path = self.gal_dir/f"{self.Name}.dll"
        return dill_path
        
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        raise NotImplementedError
        
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        #if not isinstance(other, SimPartGal):
        #    return NotImplemented
        return self._identity() == other._identity()

    def __str__(self):
        str_gal = f"Sim {self.sim}"
        str_gal = f"Gal {self.Name}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(self.N_part)} part.\nof \ntot Mass={'%.1E'%Decimal(self.M)} [M_sun]\n"
        return str_gal 
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove large attributes (if present, can be loaded again)
        for lg_att in self._large_attributes:
            state.pop(lg_att, None)
        return state

    def __setstate__(self, state):
        # Optional: restore defaults or trigger rebuild of heavy attributes
        self.__dict__.update(state)

    def _needs_unpacking(self):
        """Check whether the object is missing reconstructed attributes.
        """
        return not all(
            hasattr(self, attr)
            for attr in self._large_attributes
        )
        
    def unpack(self):
        """Public wrapper for lazy reconstruction.
        """
        if self._needs_unpacking():
            self._unpack()
        return self

    def _unpack(self):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking class...")
        raise NotImplementedError

    ########################
    ########################
    
    @property 
    def cosmo(self):
        raise NotImplementedError
        
    def run(self,reload=True):
        raise NotImplementedError
        
    def upload_prev(self,reload=True):
        if not reload:
            return False
        prev_Gal = ReadGal(self)
        if prev_Gal is False or prev_Gal != self:
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Gal.__dict__}
        return True
            
    def initialise_parts(self):
        raise NotImplementedError
        
    def verbose_assert_almost_equal(self,value1,value2=1,decimal=3,msg_title=None):
        # a verbose way of giving info if if fails
        try:
            np.testing.assert_almost_equal(value1,value2,decimal=decimal)
        except AssertionError as AssErr:
            if msg_title:
                print(msg_title)
            print("Error for \n"+str(self))
            raise AssertionError(AssErr)
        return 0
        
    def _verify_cnt(self):
        """verify that the center of mass is indeed correct
        """
        raise NotImplementedError

    def store_gal(self):
        # store this galaxy 
        with open(self.dill_path,"wb") as f:
            dill.dump(self,f)
        print(f"Saved {self.dill_path}")
        
# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(Gal,vebose=True):
    return LoadClass(path=Gal.dill_path,verbose=verbose)

def clip_coord(m,x,y,z,sigma=10):
    # clip coordinates outliers
    mask = np.ones(len(x),dtype=bool)
    for coord in x,y,z:
        mask *= np.invert(sigma_clip(coord,sigma=sigma).mask)
    return m[mask],x[mask],y[mask],z[mask]
    

