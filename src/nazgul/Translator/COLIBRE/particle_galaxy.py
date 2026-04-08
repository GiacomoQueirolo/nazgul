# copied from remade_gal.py
# adapted for COLIBRE sim.

# get random swift galaxy from get_rand_gal.py

import dill
import unyt as u  # package used by swiftsimio to provide physical units
import numpy as np
from pathlib import Path

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass,load_whatever

from nazgul.pathfinder import get_gal_dir,path_nazgul
from nazgul.Translator.particle_galaxy import BasicPartGal,store_class
from nazgul.Translator.COLIBRE import simsuite_name
from nazgul.Translator.COLIBRE.get_Gal import get_swiftgal,get_snap,get_z_snap
from nazgul.Translator.COLIBRE.get_Gal import std_sim,std_subsim,colibre_base_path
from nazgul.Translator.COLIBRE.get_Gal import min_z,max_z,min_mass,get_rnd_kw_swiftgal

def gal_path2kwGal(gal_pkl_path):
    gal_pkl_path = Path(gal_pkl_path)
    Gn_dir       = gal_pkl_path.parent.parent
    snap_dir     = Gn_dir.parent
    subsim_dir   = snap_dir.parent   
    sim_dir      = subsim_dir.parent
    Gn           = Gn_dir.name.replace("Gn","")
    snap         = snap_dir.name.replace("snap_","")
    kw_gal_full  = {}
    kw_gal_full["sim"]    = str(sim_dir.name)
    kw_gal_full["subsim"] = str(subsim_dir.name)
    
    kw_gal_full["kw_Gal"] = {"snap":str(snap),
                             "soap_index": int(Gn)}
    # M,center not necessary
    return kw_gal_full
    
# adapted from wip_select_swiftgal
def Gal2MXYZ(ColGal):
    Gal   = ColGal.swift_gal
    # Given a ColibreGal galaxy, which then plot to as swift galaxy, return Masses (in Msun) and
    # XY coords. of particles in kpc  centered around center of mass
    Mstar = Gal.stars.masses.to_physical().in_units(u.Msun)        # Msun
    Mgas  = Gal.gas.masses.to_physical().in_units(u.Msun)          # Msun
    Mdm   = Gal.dark_matter.masses.to_physical().in_units(u.Msun)  # Msun
    print("warning: using dynamical mass for bh, verify that it make sense") 
    # according to cgpt, the 2 important components are dynamical_mass and subgrid_mass
    # the first is done to compute the potential, the other is updated w. the accretion and used for feedback calc.
    # -> should be correct to use dynamical mass
    Mbh   = Gal.black_holes.dynamical_masses.to_physical().in_units(u.Msun)  # Msun
    Ms    = np.concatenate([Mstar,Mgas,Mdm,Mbh]) #Msun
    
    # Particle pos
    Xstar,Ystar,Zstar =  np.transpose(Gal.stars.coordinates.to_physical().in_units(u.kpc))        # kpc
    Xgas,Ygas,Zgas    =  np.transpose(Gal.gas.coordinates.to_physical().in_units(u.kpc))          # kpc
    Xdm,Ydm,Zdm       =  np.transpose(Gal.dark_matter.coordinates.to_physical().in_units(u.kpc))  # kpc
    Xbh,Ybh,Zbh       =  np.transpose(Gal.black_holes.coordinates.to_physical().in_units(u.kpc))  # kpc
    Xs = np.concatenate([Xstar,Xgas,Xdm,Xbh]) #kpc
    Ys = np.concatenate([Ystar,Ygas,Ydm,Ybh]) #kpc
    Zs = np.concatenate([Zstar,Zgas,Zdm,Zbh]) #kpc
    X_cm = np.sum(Xs*Ms)/np.sum(Ms)
    Y_cm = np.sum(Ys*Ms)/np.sum(Ms)
    Z_cm = np.sum(Zs*Ms)/np.sum(Ms)
    
    Xs-=X_cm
    Ys-=Y_cm
    Zs-=Z_cm
    
    return Ms, Xs,Ys,Zs

def get_kw_SimPartGal(kw_Gal,sim,simsuite,subsim,data_dir,z,snap,M,Centre,reload):
    assert simsuite==simsuite_name
    return {"kw_Gal":kw_Gal,"sim":sim,"subsim":subsim}

# basically a wrapper for swift galaxies
class SimPartGal(BasicPartGal):
    # what we need is
    # z
    # cosmology
    # particles (get from Gal2MXYZ)
    # identity
    # name
    
    # define name to verify identity
    _type_id = "SimPartGal_"+simsuite_name
    _large_attributes_setup  = ["stars","gas","dark_matter","black_holes","swift_gal","_swift_gal"]
    _large_attributes_unpack = []
    
    simsuite = simsuite_name
    def __init__(self,kw_Gal,sim=std_sim,subsim=std_subsim):
        #kw_Gal: soap_index,snap and/or z
        #self.kw_Gal     = kw_Gal
        self.soap_index  = kw_Gal["soap_index"] #self.swift_gal.halo_catalogue.soap_index
        self.z,self.snap = get_z_snap(z=kw_Gal.get("z",None),
                            snap=kw_Gal.get("snap",None))
        
        self.sim         = Path(sim)
        self.subsim      = Path(subsim)
        
        # here we load but not store the swift galaxy to avoid 
        # increasing the memory load
        self.soap_file  = Path(self.swift_gal.halo_catalogue.soap_file)
        #'/cosma8/data/dp004/colibre/Runs/L0025N0752/THERMAL_AGN_m5/SOAP-HBT/halo_properties_0127.hdf5'
        
        self.a =  self.swift_gal.metadata.a
        self.verbose_assert_almost_equal((1/self.a)-1,self.z,msg_title="Redshifts")
        self.verify_snap()

        self.gal_dir  = get_gal_dir(kw_Gal,snap=self.snap,
                                    sim=self.sim,subsim=self.subsim,
                                    simsuite=self.simsuite)
        mkdir(self.gal_dir)
        
        # total mass
        SphOverDens = self.swift_gal.halo_catalogue.spherical_overdensity_500_crit
        self.M      = SphOverDens.total_mass.to_comoving_value("Msun")[0] #Msun
        # coord of the centre
        self.centre = self.swift_gal.centre.to_comoving_value("Mpc") 
        #self.part_dir = get_part_dir(self.snap,data_dir=data_dir,**kw_sim)
       
        
        
    @property
    def swift_gal(self):
        try:
            return self._swift_gal
        except AttributeError:
            # only compute it once
            swift_gal = get_swiftgal(sim=self.sim,
                                     subsim=self.subsim,
                                     snap=self.snap,
                                     soap_index=self.soap_index)
            self._swift_gal = swift_gal
            return swift_gal
            
    def initialise_parts(self):
        # heavy -> avoid until necessary and do not store
        sg               = self.swift_gal
        self.stars       = sg.stars
        self.gas         = sg.gas
        self.dark_matter = sg.dark_matter
        self.black_holes = sg.black_holes
        self.N_part = len(sg.gas.particle_ids) +\
                 len(sg.dark_matter.particle_ids) +\
                 len(sg.stars.particle_ids) +\
                 len(sg.black_holes.particle_ids) 
        return 0
        
    @property
    def cosmo(self):
        return self.swift_gal.metadata.cosmology
    
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self._type_id,self.sim,self.subsim,
            self.snap,self.soap_index)
        return Id 
        
    def upload_prev(self,verbose=True):
        prev_Gal = ReadGal(self)
        if prev_Gal is False:
            if verbose:
                print("Failed loading of prev. gal.")
            return False
        if prev_Gal != self:
            if verbose:
                print(f"Prev. Gal not equal to self: {prev_Gal._identity()==self._identity()}")
                print(f"Prev. Gal: {prev_Gal._identity()}")
                print(f"Self:      {self._identity()}")
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Gal.__dict__}
        if verbose:
            print("Loaded prev. gal.")
        return True
    
    def store_gal(self):
        # store class instance 
        store_class(self,path=self.dill_path)

    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------
    def _setup(self):
        """Setup all attributes NEEDED FOR COMPUTATION
        that were intentionally removed before serialization.
        """
        print("Unpacking Particle Galaxy ...")
        self.swift_gal
        self.initialise_parts()
        print("... unpacked Particle Galaxy")
        return 
        
    def _unpack(self):
        """Reconstruct attributes AFTER COMPUTATION
        that were intentionally removed before serialization.
        """
        # there is nothing to do for this class
        return 

    ########################     
    @property
    def name(self):
        # arbitrary funct to give name to gal
        # assuming that the simulation stays ~ const
        nm = f"G{self.soap_index}"
        return nm
        
    def verify_snap(self):
        # quick validity check that the snap is correct
        nm_file        = str(self.soap_file.name)
        snap_from_file = nm_file.split("_")[-1].split(".")[0]
        assert self.snap==snap_from_file
        
    def run(self,reload=True):
        upload_successful = False
        if reload:
            upload_successful = self.upload_prev(verbose=True)
        self.setup()
        if not upload_successful:
            self.store_gal()
            
# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(Gal,verbose=True):
    if not Path(Gal.dill_path).is_file():
        return False
    other_Gal = LoadClass(path=Gal.dill_path,verbose=verbose,path_base=path_nazgul)
    # If failed, return False
    if not other_Gal: 
        if verbose:
            print("Failed loading of prev.")       
        return False
    # check that loaded Gal would be indeed the same
    if Gal!=other_Gal:
        if verbose:
                print(f"Prev. Gal not equal to self: {other_Gal._identity()==Gal._identity()}")
                print(f"Prev. Gal: {other_Gal._identity()}")
                print(f"Self:      {Gal._identity()}")
        return False
    return other_Gal

def LoadGal(path,if_fail_recompute=True,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose,path_base=path_nazgul)
    if not Gal and if_fail_recompute:
        full_kwgal = gal_path2kwGal(path)
        Gal        = SimPartGal(**full_kwgal)
    if Gal:
        Gal.unpack()
    return Gal
    
def get_rnd_SPG(sim=std_sim,subsim=std_subsim,
               colibre_base_path=colibre_base_path,
               min_mass= min_mass,
               min_z=min_z,
               max_z=max_z
              ):
    """Randomly sample a galaxy from the simulation 
    """
    kw_swiftgal = get_rnd_kw_swiftgal(colibre_base_path=colibre_base_path,
                            sim=sim,
                            subsim=subsim,
                            max_z=max_z,
                            min_z=min_z,
                            min_mass=min_mass)
    kw_Gal = {"soap_index":kw_swiftgal["soap_index"],
              "snap":kw_swiftgal["snap_str"]}
    SPG    = SimPartGal(kw_Gal=kw_Gal,
                       sim=sim,
                       subsim=subsim)
    return SPG

def get_all_SPG(sim=std_sim,subsim=std_subsim,
               colibre_base_path=colibre_base_path,
               min_mass= min_mass,
               min_z=min_z,
               max_z=max_z,
               limit_n=1e3
               ):
    """Get all possible galaxies in the range"""
    raise NotImplementedError("Still to write up")