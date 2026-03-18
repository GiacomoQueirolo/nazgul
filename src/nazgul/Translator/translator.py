# interface between nazgul and the simulation-specific "translators"
import numpy as np
from pathlib import Path
from importlib import import_module

from python_tools.get_res import LoadClass
from python_tools.get_res import load_whatever

from nazgul.pathfinder import std_data_dir
# allowed simulations suites
from nazgul.Translator import SimSuiteNames
from nazgul.Translator import std_simsuite,std_sim,min_z,max_z,min_mass
from nazgul.pathfinder import get_part_dir,get_gal_dir
from nazgul.Translator.pathfinder import translate_galname

def check_simsuite(simsuite):
    if simsuite not in SimSuiteNames:
        raise RuntimeError(f"The simsuite {simsuite} is not yet implemented, allowed only {SimSuiteNames}") 
        
def get_sim_func(simsuite,func_name):
    check_simsuite(simsuite)
    module_path = f"nazgul.Translator.{simsuite}.particle_galaxy"
    module = import_module(module_path)
    func   = getattr(module, func_name)
    return func

# useful function (but find better place)
def get_CM(Ms,Xs,Ys,Zs=None):
    """Get Center of Mass (CM)
    """
    X_cm = np.sum(Xs*Ms)/np.sum(Ms)
    Y_cm = np.sum(Ys*Ms)/np.sum(Ms)
    if Zs is None:
        return X_cm,Y_cm
    else:
        Z_cm = np.sum(Zs*Ms)/np.sum(Ms)
        return X_cm,Y_cm,Z_cm


def get_rnd_kw_gal(simsuite=std_simsuite,sim=std_sim,subsim=None,
                   min_mass = str(min_mass),
                   min_z    = str(min_z),
                   max_z    = str(max_z),
                   check_prev=True,save_pkl=True):
    """Given the simulation, the range of redshift and minimum mass required, 
        returns a kwargs of a random galaxy from the simulation
    """
    check_simsuite(simsuite)
    if simsuite=="EAGLE":
        from nazgul.Translator.EAGLE.particle_galaxy import get_rnd_kw_gal
        min_mass = str(min_mass)
        min_z    = str(min_z)
        max_z    = str(max_z)
        kw     = get_rnd_kw_gal(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,\
                        check_prev=check_prev,plot=False,save_pkl=save_pkl)
    elif simsuite=="COLIBRE":
        from nazgul.Translator.COLIBRE.particle_galaxy import get_rnd_kw_gal
        simulation_dir = Path(sim)/subsim
        kw             = get_rnd_kw_gal(simulation_dir=simulation_dir,
                                min_mass=min_mass,max_z=max_z,min_z=min_z)
    return kw

def get_z_snap(simsuite,z=None,snap=None):
    get_z_snap = get_sim_func(simsuite,"get_z_snap")
    return get_z_snap(z=z,snap=snap)

def Gal2MXYZ(Gal):
    Gal.run()
    simsuite = Gal.simsuite
    Gal2MXYZ = get_sim_func(simsuite,"Gal2MXYZ")
    return Gal2MXYZ(Gal) 

def Gal2kwMXYZ(Gal): 
    Ms, Xs,Ys,Zs = Gal2MXYZ(Gal)
    return {"Ms":Ms,"Xs":Xs,"Ys":Ys,"Zs":Zs}
    
# from path to kw of Gal

def gal_path2simsuite(gal_dill_path,data_dir=std_data_dir):
    simsuite       = Path(gal_dill_path).parts[1]
    return simsuite
    
def gal_path2kwGal(gal_dill_path,data_dir=std_data_dir):
    """From path extract the required inputs for PartGal class
    """
    gal_dill_path   = Path(gal_dill_path)
    simsuite        = gal_path2simsuite(gal_dill_path,data_dir)
    gal_path2kwGal  = get_sim_func(simsuite,"gal_path2kwGal")
    kw_Gal_full     = gal_path2kwGal(gal_dill_path)
    return kw_Gal_full

def get_rnd_PG(simsuite,**kw_galpart):
    get_rnd_SPG = get_sim_func(simsuite,"get_rnd_SPG")
    SPG = get_rnd_SPG(**kw_galpart)
    PG  = SPG2PG(simsuite,SPG)
    return PG
    
def LoadGal(path,if_fail_recompute=True,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose)
    if not Gal and if_fail_recompute:
        print("Recomputing ...")
        simsuite    = gal_path2simsuite(path)
        kw_Gal_full = gal_path2kwGal(path)
        Gal         = PartGal(simsuite=simsuite,**kw_Gal_full)
    return Gal

def SPG2PG(simsuite,SPG):
    return PartGal.from_SimPartGal(simsuite=simsuite,SPG=SPG)

class PartGal():
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units

    -> a wrapper of the individual SimPartGal of the specific ones for each simulation suite
    """
    def __init__(self, 
                 kw_Gal,# way to ID the galaxy (sim-dependent)
                 simsuite=std_simsuite, # identity of the simulation
                 sim=std_sim, # which sim. between the sim. suite
                 subsim=None, # optional sub. dir of simulation
                 data_dir=std_data_dir, # where is the particle data stored
                 z=None,snap=None,    # redshift or snap
                 M=None,Centre=None, # these can be recovered
                 reload=True,     # set to false only for debug
                ):
        
        check_simsuite(simsuite)
        self.simsuite     = simsuite
        SimPartGal        = get_sim_func(self.simsuite,"SimPartGal")
        get_kw_SimPartGal = get_sim_func(self.simsuite,"get_kw_SimPartGal")
        kw_SimPartGal     = get_kw_SimPartGal(kw_Gal=kw_Gal,simsuite=simsuite, 
                                          sim=sim,subsim=subsim,
                                          data_dir=data_dir,
                                          z=z,snap=snap,
                                          M=M,Centre=Centre,
                                          reload=reload)
        self._SimPartGal = SimPartGal(**kw_SimPartGal)
        #self._SimPartGal.run(reload=reload)

    @classmethod
    def from_SimPartGal(cls, simsuite, SPG):
        """Construct from an existing SimPartGal instance."""
        obj = cls.__new__(cls)  # bypass __init__
        obj.simsuite = simsuite
        obj._SimPartGal = SPG
        return obj

    def __getattr__(self, name):
        try:
            return getattr(self._SimPartGal, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")
"""
        self.sim      = sim
        self.subsim   = subsim
        self.simsuite = simsuite
        kw_sim        = {"sim":sim,"subsim":subsim,"simsuite":simsuite}
        z,snap        = get_z_snap(z=z,snap=snap,simsuite=simsuite)
        self.snap     = snap
        self.z        = z
        self.Name     = translate_galname(kw_Gal=kw_Gal,simsuite=simsuite) 
        # Input Dir:
        self.part_dir = get_part_dir(snap,data_dir=data_dir,**kw_sim)
        # Output dir:
        self.gal_dir  = get_gal_dir(kw_Gal,snap=snap,data_dir=data_dir,**kw_sim)# data and simsuite are for now fixed

        
        # Mass and Centre can be recovered
        kw_MCntr      = get_kwMCntr(kw_Gal,z=z,**kw_sim)
        _M            = kw_MCntr["M"]
        _Centre       = kw_MCntr["Centre"]
        if M is not None:
            assert M  == _M
        if Centre is not None:
            assert np.all(Centre == _Centre)
        self.M        = _M
        self.centre   = _Centre

        
        self.a,self.h,self.boxsize = self.read_snap_header()

        # all paths are dealt as properties
        mkdir(self.gal_dir)        
        self.run(reload=reload)
"""
