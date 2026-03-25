"""
-> this has to be generalised for any simulation, for now only adapted to EAGLE
Define the galaxy class PartGal (storing all particle data), as well as related helper functions, e.g. to sample galaxies 
"""

import glob
import dill
import h5py
import numpy as np
from pathlib import Path
import astropy.units as u
from decimal import Decimal
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from functools import cached_property
from multiprocessing import Pool,cpu_count

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass
from python_tools.get_res import load_whatever
from astropy.cosmology import FlatLambdaCDM

from nazgul.Translator.EAGLE.get_gal_indexes import get_gals
from nazgul.Translator.EAGLE.get_gal_indexes import get_catpath
from nazgul.Translator.EAGLE.fnct import _count_part,_mass_part
from nazgul.Translator.EAGLE.fnct import get_z_snap,read_snap_header
from nazgul.pathfinder import get_gal_dir,get_part_dir,std_sim,std_simsuite,std_data_dir,path_nazgul
from nazgul.Translator.EAGLE.fnct import _count_part,_mass_part
from nazgul.Translator.EAGLE.fnct import get_z_snap,read_snap_header

from nazgul.Translator.translator import min_z,max_z,min_mass

from nazgul.Translator.particle_galaxy import BasicPartGal,clip_coord

from nazgul.Translator.EAGLE.pathfinder import simsuite_name

def gal_path2kwGal(gal_path):
    """From path extract the ALL required inputs for SimPartGal class
    """
    gal_path     = Path(gal_path)

    GnSGn_dir    = gal_path.parent.parent
    snap_dir     = GnSGn_dir.parent
    sim_dir      = snap_dir.parent
    #simsuite_dir = sim_dir.parent
    _Gn,SGn      = GnSGn_dir.name.split("SGn")
    Gn           = _Gn.replace("Gn","")
    snap         = snap_dir.name.replace("snap_","").lstrip("0")
    kw_gal_full  = {}
    kw_gal_full["kw_Gal"] = {"Gn":int(Gn),
                             "SGn": int(SGn)}
    kw_gal_full["snap"]   = str(snap)
    kw_gal_full["sim"]    = str(sim_dir.name)
    # M,center not necessary
    return kw_gal_full

def get_rnd_gal_indexes(sim=std_sim,
                        min_mass = str(min_mass),
                        min_z=str(min_z),
                        max_z=str(max_z),
                        check_prev=True,save_pkl=True):
    """Given the simulation, the range of redshift and minimum mass required, 
        returns a random galaxy from the simulation
    """
    min_mass = str(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    data     = get_gals(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,\
                        check_prev=check_prev,plot=False,save_pkl=save_pkl)
    index = np.arange(len(data["z"]))
    rnd_i = np.random.choice(index)
    kw = {}
    for k in data.keys():
        if k=="query" or k=="sim":
            kw[k] = data[k]
        else:
            kw[k] = data[k][rnd_i]
    return kw


def get_kw_SimPartGal(kw_Gal,sim,simsuite,subsim,data_dir,z,snap,M,Centre,reload):
    
    assert simsuite==simsuite_name

    return {"kw_Gal":kw_Gal,"sim":sim,"snap":snap,"z":z,"M":M,"Centre":Centre}

# index for particle types:
# gas,dm, stars,bh : 0,1,4,5
class SimPartGal(BasicPartGal):
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units
    """
    _large_attributes = ["stars","gas","dm","bh"]
    # indexes of particles: gas,dm,stars,bh:
    indexes = [0,1,4,5]
    # n* of files per snapshot:
    nfiles  = 16

    def __init__(self, 
                 kw_Gal, # identity of the galaxy: Gn,SGn
                 sim=std_sim, 
                 data_dir=std_data_dir,
                 z=None,snap=None, # define redshift bin
                 M=None,Centre=None # these can be recovered
                 ):   
        self.sim      = sim
        z,snap        = get_z_snap(z,snap)
        self.snap     = snap
        self.z        = z
        self.Gn       = kw_Gal["Gn"]
        self.SGn      = kw_Gal["SGn"]
        # Input Dir:
        self.part_dir = get_part_dir(snap,sim=sim,simsuite=simsuite_name,data_dir=data_dir)
        # Output dir:
        self.gal_dir  = get_gal_dir(kw_gal=kw_Gal,snap=snap,sim=sim,
                                    simsuite=simsuite_name,data_dir=data_dir)# data and simsuite are for now fixed
        # Mass and Centre can be recovered
        kw_MCntr      = get_kwMCntr(self.Gn,self.SGn,z=z,sim=sim)
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
        #self.run(reload=reload)
    @property
    def Name(self):
        #Note: this is unique only within the snap
        return f"Gn{self.Gn}SGn{self.SGn}"
                
    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self.sim,self.snap,self.Name)
        return Id
        
    def __str__(self):
        str_gal = f"Gal {self.Gn}.{self.SGn}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(int(self.N_part))} part.\nof \ntot Mass={'%.1E'%Decimal(float(self.M))} [M_sun]\n"
        str_gal +=f" divided in N \n\
                Stars:{'%.1E'%Decimal(int(self.N_stars))}\n\
                Gas:{'%.1E'%Decimal(int(self.N_gas))}\n\
                DM:{'%.1E'%Decimal(int(self.N_dm))}\n\
                BH:{'%.1E'%Decimal(int(self.N_bh))}\n"
        str_gal +=f"and Mass in \n\
                    Stars:{'%.1E'%Decimal(float(self.M_stars))} [M_sun]\n\
                    Gas:{'%.1E'%Decimal(float(self.M_gas))} [M_sun]\n\
                    DM:{'%.1E'%Decimal(float(self.M_dm))} [M_sun]\n\
                    BH:{'%.1E'%Decimal(float(self.M_bh))} [M_sun]\n"
        return str_gal 

    # ------------------------------------------------------------------
    # Lazy reconstruction logic
    # ------------------------------------------------------------------

    def _unpack(self):
        """Reconstruct all attributes that were intentionally removed
        before serialization.
        """
        print("Unpacking class...")
        self.initialise_parts()

    ########################
    ########################
    
    @property 
    def cosmo(self):
        return FlatLambdaCDM(H0=self.h*100, Om0=1-self.h)

    def run(self,reload=True):
        upload_successful = False
        if reload:
            upload_successful = self.upload_prev(reload=reload)
        if not upload_successful:
            # useful to check Center of Mass
            self.xy_propr2comov = self.prop2comov("Coordinates") 
            self.m_propr2comov  = self.prop2comov("Mass") 
            # actually store the gal
            self.initialise_parts()
            self._count_tot_part()
            self._mass_tot_part()
            self._verify_cnt()
            self.store_gal()

    def upload_prev(self,reload=True):
        if not reload:
            return False
        prev_Gal = ReadGal(self)
        if prev_Gal is False or prev_Gal != self:
            return False
        # if common attribute, they are overwritten by previous:
        self.__dict__ = {**self.__dict__,**prev_Gal.__dict__}
        if not getattr(self,"stars",False):
            self.initialise_parts()
        return True
            
    
    def prop2comov(self,varType):
        """Proper coordinate/mass to comoving
        physically motivated 
        """
        if varType=="Coordinates":
            aexp = 1
        elif varType=="Mass":
            aexp = 0
        return  1/(self.a**aexp)

    def read_snap_header(self):
        return read_snap_header(z=self.z,snap=self.snap,sim=self.sim)

    def initialise_parts(self):
        self.gas   = self.read_part(0)
        self.dm    = self.read_part(1)
        self.stars = self.read_part(4)
        self.bh    = self.read_part(5)
        return 0
        
    ## Loading particles #
    ######################
    
    def _get_files(self):
        files = [glob.glob(f"{self.part_dir}/snap*.{i}.hdf5")[0] for i in range(self.nfiles)]
        if len(files) != self.nfiles:
            raise RuntimeError(f"Expected {self.nfiles} files, got {len(files)}")
        return files

    def build_index(self, itype):
        """
        Return mapping: {file_id: particle_indices}
        """
        files = self._get_files()
        index_map = {}

        for i, fl in enumerate(files):
            with h5py.File(fl, "r") as f:
                grp = f[f"PartType{itype}/GroupNumber"][:]
                sub = f[f"PartType{itype}/SubGroupNumber"][:]
                
                mask = np.logical_and(grp==self.Gn,sub==self.SGn)
                idx  = np.where(mask)[0]
                
                if len(idx) > 0:
                    # sort them to speed up the hdf5 readout:
                    idx = np.sort(idx)
                    index_map[i] = idx
        return index_map
        
    @cached_property
    def kw_index_maps(self):
        index_maps = {f'{itype}':self.build_index(itype) for itype in self.indexes}
        return index_maps
        
    def read_part(self, itype):
        """
        Load only selected particles using index_map
        """
        files = self._get_files()
        tasks = [(files[i], idx, itype,self.boxsize,self.centre) for i, idx in self.kw_index_maps[str(itype)].items()]

        nproc = min(cpu_count(), len(files))

        with Pool(nproc) as pool:
            results = pool.map(_load_one_file, tasks)

        # ---- merge ----
        output = {"coords":[],"mass":[]}
        if itype!=1:
            output["smooth"] = []
        for r in results:
            output["coords"].append(r["coords"])
            output["mass"].append(r["mass"])
            if "smooth" in r:
                output["smooth"].append(r["smooth"])
        
        output["coords"] = np.vstack(output["coords"])
        output["mass"]   = np.hstack(output["mass"])
        if itype != 1:
            output["smooth"] = np.hstack(output["smooth"])
            
        return output
        
    def _count_tot_part(self):
        """Count total particle number"""
        self.N_gas      = _count_part(self.gas)
        self.N_dm       = _count_part(self.dm)
        self.N_stars    = _count_part(self.stars)
        self.N_bh       = _count_part(self.bh)
        self.N_part     =  self.N_gas+self.N_dm+self.N_stars+self.N_bh
        return self.N_part

    def _mass_tot_part(self):
        """Count total particle mass and verify it's the same as 
        the input one
        """
        self.M_gas   = _mass_part(self.gas)
        self.M_dm    = _mass_part(self.dm)
        self.M_stars = _mass_part(self.stars)
        self.M_bh    = _mass_part(self.bh)
        
        self.M_tot   =  self.M_gas+self.M_dm+self.M_stars +self.M_bh
        self.verbose_assert_almost_equal(float(self.M_tot)/float(self.M),1,decimal=3,msg_title="The summed mass and the expected mass differs")
        return self.M_tot
                
    def _verify_cnt(self):
        """verify that the center of mass is indeed correct
        """
        if not hasattr(self,"M_tot"):
            self._mass_tot_part()
        # conv. in comov. coordinates
        xy_dm  = self.dm["coords"]*self.xy_propr2comov
        xy_gas = self.gas["coords"]*self.xy_propr2comov
        xy_st  = self.stars["coords"]*self.xy_propr2comov
        xy_bh  = self.bh["coords"]*self.xy_propr2comov

        # m in comov coord
        m_dm  = self.m_propr2comov*np.broadcast_to(self.dm["mass"],(3,self.N_dm)).T
        m_gas = self.m_propr2comov*np.broadcast_to(self.gas["mass"],(3,self.N_gas)).T
        m_st  = self.m_propr2comov*np.broadcast_to(self.stars["mass"],(3,self.N_stars)).T
        m_bh  = self.m_propr2comov*np.broadcast_to(self.bh["mass"],(3,self.N_bh)).T
        
        cm_dm   = np.sum(xy_dm*m_dm,axis=0)
        cm_gs   = np.sum(xy_gas*m_gas,axis=0)
        cm_st   = np.sum(xy_st*m_st,axis=0)
        cm_bh   = np.sum(xy_bh*m_bh,axis=0)

        cnt_m  = (cm_dm+cm_gs+cm_st+cm_bh)/(self.M*self.m_propr2comov)
 
        center_actual  = np.array(cnt_m)
        center_desired = self.centre
        #np.testing.assert_almost_equal(center_desired,center_desired,decimal=2)
        self.verbose_assert_almost_equal(center_desired,center_desired,decimal=2,msg_title="Centre") 

def _load_one_file(args):
    fl, indices, itype,boxsize,centre = args
    results = {}
    with h5py.File(fl, "r") as f:
        # Coordinates
        #############
        splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        
        _coords = []
        for chunk in splits:
            start, end = chunk[0], chunk[-1] + 1
            data = f[f"PartType{itype}/Coordinates"][start:end]
            _coords.append(data[chunk - start])
        coords = np.vstack(_coords)
        # conversion
        cgs  = f[f"PartType{itype}/Coordinates"].attrs["CGSConversionFactor"]
        aexp = f[f"PartType{itype}/Coordinates"].attrs["aexp-scale-exponent"]
        hexp = f[f"PartType{itype}/Coordinates"].attrs["h-scale-exponent"]
        a    = f["Header"].attrs["Time"]
        h    = f["Header"].attrs["HubbleParam"]

        coords = coords * cgs * (a ** aexp) * (h ** hexp)*u.cm.to(u.Mpc)
        # Periodic wrap coordinates around centre.
        boxsize           = boxsize*(h**hexp)
        centre            = centre*(a**aexp) # given in comoving (but not 1/h)
        results['coords'] = np.mod(coords-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize       
        
        # Mass
        ######
        if itype != 1:
            _mass2scale = []
            for chunk in splits:
                start, end = chunk[0], chunk[-1] + 1
                data = f[f"PartType{itype}/Mass"][start:end]
                _mass2scale.append(data[chunk - start])
            mass2scale = np.hstack(_mass2scale)
            cgs  = f[f"PartType{itype}/Mass"].attrs["CGSConversionFactor"]
            aexp = f[f"PartType{itype}/Mass"].attrs["aexp-scale-exponent"]
            hexp = f[f"PartType{itype}/Mass"].attrs["h-scale-exponent"]
            mass = mass2scale * cgs * (a ** aexp) * (h ** hexp)*u.g.to(u.Msun)
            results["mass"] = mass
            # Smoothness
            ############
            _smooth2scale = []
            for chunk in splits:
                start, end = chunk[0], chunk[-1] + 1
                data = f[f"PartType{itype}/SmoothingLength"][start:end]
                _smooth2scale.append(data[chunk - start])
            smooth2scale = np.hstack(_smooth2scale)

            smooth = smooth2scale * cgs * (a ** aexp) * (h ** hexp)*u.cm.to(u.Mpc)
            results["smooth"] = smooth
        else:
            dm_mass = f['Header'].attrs.get('MassTable')[1]
            n_particles = f['Header'].attrs.get('NumPart_Total')[1]
            # Create an array of lenght n_particles each set to dm_mass
            print("len(indices),n_particles",len(indices),n_particles)
            mass2scale = np.ones(len(indices), dtype='f8') * dm_mass
            # Use the conversion factors from the mass entry in the gas particles.
            cgs  = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
            aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
            hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
            # Convert to proper/physical mass 
            results["mass"] = np.multiply(mass2scale, cgs*(a**aexp)*(h**hexp), dtype='f8')*u.g.to(u.Msun)
 
    return results

# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(Gal,verbose=True):
    if not Path(Gal.dill_path).is_file():
        return False
    return LoadGal(path=Gal.dill_path,verbose=verbose)

def LoadGal(path,if_fail_recompute=True,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose,path_base=path_nazgul)
    if not Gal and if_fail_recompute:
        kwGal   = gal_path2kwGal(path)
        Gal     = SimPartGal(**kwGal)
    if Gal:
        Gal.unpack()
    return Gal

# to simplify the input: given the sim, z, and GnSgn, 
# we get the mass and center of the galaxy for input of SimPartGal 
def get_myCat(Gn,SGn,z,sim,min_mass=min_mass,dz=0.05):
    min_z=str(z-0.05)
    max_z=str(z+0.05)

    # try first finding the exact one
    cat_path = get_catpath(min_mass=min_mass,\
                           min_z=min_z,max_z=max_z,
                           sim=sim)
    if Path.is_file(cat_path):
        myCat = load_whatever(cat_path)
    else:
        # try all the other
        found = False
        # now quite a desperate attempt
        for pkl in Path(cat_path).parent.glob("*.pkl"):
            cat = load_whatever(pkl)
            if np.any(np.abs(cat["z"]-z)<dz):
                zz      = min(cat["z"], key=lambda x:abs(x-z))
                indexx  = np.where((cat["Gn"]==Gn) & (cat["SGn"]==SGn) & (cat["z"]==zz))[0]
                if len(indexx)==1:
                    found  = True
                    myCat = cat
                    index  = indexx[0]
                    break
        if not found:
            # get_gals is slow-ish but safe (require internet access!)
            # safe-ish since we now hard write the min_mass - but should be fine
            # TODO: find a safer approach
    
            myCat   = get_gals(sim=sim,
                                min_mass=min_mass,min_z=min_z,max_z=max_z,
                                save_pkl=False,check_prev=True,verbose=False,plot=False)
    
    z        = min(myCat["z"], key=lambda x:abs(x-z))
    index    = np.where((myCat["Gn"]==Gn) & (myCat["SGn"]==SGn) & (myCat["z"]==z))[0]
    if len(index)>1:
        raise RuntimeError(f"Found multiple galaxies with same z, G and SGn")
    index    = index[0] 
    return myCat,index
    
def get_kwMCntr(Gn,SGn,sim=std_sim,
                z=None,snap=None):
    z,snap       = get_z_snap(z,snap)
    myCat,index  = get_myCat(Gn,SGn,z,sim)
    Centre       = np.array([myCat["CMx"],myCat["CMy"],myCat["CMz"]]).T[index]
    kwMCntr      = {"M":myCat["M"][index],
                    "Centre":Centre}
    return kwMCntr

def get_rnd_SPG(sim=std_sim,min_mass=min_mass,min_z=min_z,max_z=max_z,
               check_prev=True,save_pkl=True):
    """Randomly sample a galaxy from the simulation 
    """

    kw_gal   = get_rnd_gal_indexes(sim=sim,min_mass=min_mass,
                                   min_z=min_z,max_z=max_z,
                                   check_prev=check_prev,save_pkl=save_pkl)
    z        = kw_gal["z"] 
    M        = kw_gal["M"] 
    kw_Gal   = {"Gn":kw_gal["Gn"],"SGn":kw_gal["SGn"]}
    Centre   = np.array([kw_gal["CMx"],kw_gal["CMy"],kw_gal["CMz"]]) 
    
    kwGal    = {"z":z,"kw_Gal":kw_Gal,"sim":sim,"M":M,"Centre":Centre}
    SPG      = SimPartGal(**kwGal)
    return SPG


def Gal2MXYZ(Gal): 
    """Given the galaxy, return Masses (in Msun) and
    XY coords. of particles in kpc  centered around center
    """
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
    Xs = np.concatenate([Xstar,Xgas,Xdm,Xbh])*u.Mpc.to("kpc")*u.kpc #kpc
    Ys = np.concatenate([Ystar,Ygas,Ydm,Ybh])*u.Mpc.to("kpc")*u.kpc #kpc
    Zs = np.concatenate([Zstar,Zgas,Zdm,Zbh])*u.Mpc.to("kpc")*u.kpc #kpc

    # clip particle outliers
    Ms,Xs,Ys,Zs = clip_coord(Ms,Xs,Ys,Zs)
    
    # center around the center of the galaxy
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 
    # ->  it's given in cMpc (not cMpc/h) fsr
    Cx,Cy,Cz = Gal.centre*u.Mpc.to("kpc")*u.kpc/(Gal.xy_propr2comov) # (now) kpc 
    
    Xs -= Cx
    Ys -= Cy
    Zs -= Cz
    return Ms, Xs,Ys,Zs

# The following should be done in the test_particle_galaxy
"""    
# for debug:
if __name__=="__main__":

    SPG = get_rnd_SPG()
    fig, ax = plt.subplots(3)
    nx = 100
    for name,part in zip(["stars","dm","gas"],[SPG.stars,SPG.dm,SPG.gas]):
        coords = part["coords"]
        x,y,z  = coords.T
        print(np.std(coords,axis=0))
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[0].set_xlabel("X [kpc]")
    ax[2].set_xlabel("Z [kpc]")
    ax[1].set_xlabel("Y [kpc]")
    ax[2].legend()
    namefig = f"tmp/hist_by_hand_parts.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close(fig)
    print(f"Saved {namefig}") 
    
    print("Testing dens map")
    from test_proj_part_hist import get_dens_map_rotate_hist
    NG.proj_dir = "./tmp/proj_part"
    mkdir(NG.proj_dir)
    dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=NG,pixel_num=100j,
                                                            z_source_max=5,verbose=True,plot=True)
"""
