"""
Define the galaxy class PartGal (storing all particle data), as well as related helper functions, e.g. to sample galaxies 
"""

import os
import glob
import dill
import h5py
import numpy as np
from pathlib import Path
import astropy.units as u
from decimal import Decimal
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from python_tools.tools import mkdir
from python_tools.get_res import LoadClass
from python_tools.get_res import load_whatever
from astropy.cosmology import FlatLambdaCDM

from get_gal_indexes import get_gals
from get_gal_indexes import get_catpath
from fnct import std_gal_dir,sim2galdir,galdir2sim,_count_part,_mass_part
from fnct import part_data_path,std_sim,get_z_snap,prepend_str,get_snap,read_snap_header

# combination btw get_rnd_gal and _get_rnd_gal
z_source_max = 4
verbose      = True
min_z        = 0.02
max_z        = 2
# max_z is implicitely =3.53
min_mass     = 1e12 # Sol Mass
def get_rnd_gal_indexes(sim=std_sim,min_mass = str(min_mass),min_z=str(min_z),
                        max_z=str(max_z),check_prev=True,save_pkl=True):
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

# from path to kw of Gal
def gal_path2kwGal(gal_pkl_path,gal_dir=std_gal_dir):
    """From path extract the required inputs for PartGal class
    """
    gal_pkl_path = Path(gal_pkl_path)
    kw_gal       = {} 
    str_snap     = gal_pkl_path.parent.name
    snap         = str_snap.replace("snap_","")
    str_gal_name = gal_pkl_path.name
    gal_name     = str_gal_name.replace(".pkl","")
    sGn,SGn      = gal_name.split("SGn")
    Gn           = sGn.replace("Gn","")
    kw_gal["Gn"]   = int(Gn)
    kw_gal["SGn"]  = int(SGn)
    kw_gal["snap"] = str(snap)
    kw_gal["sim"]  = str(galdir2sim(gal_dir))
    # M,center not necessary
    return kw_gal
    
# index for particle types:
# gas,dm, stars,bh : 0,1,4,5 
class PartGal:
    """Given the simulation, snap (or z) and galaxy numbers, set up a class
    with all the needed particle properties converted in physical units
    """
    def __init__(self, 
                 Gn, SGn,sim=std_sim, # identity of the galaxy
                 z=None,snap=None,    # redshift or snap
                 part_dir=part_data_path, # where are stored the particles
                 gal_dir=std_gal_dir,     # where to store it
                 M=None,Centre=None, # these can be recovered
                 reload=True):    # set to false only for debug
        self.sim      = sim
        z,snap        = get_z_snap(z,snap)
        self.snap     = snap
        self.z        = z
        self.Gn       = Gn
        self.SGn      = SGn
        #Note: this is unique only within the snap
        self.Name     = f"Gn{self.Gn}SGn{self.SGn}" 
        self.gal_dir  = Path(gal_dir)
        self.part_dir = Path(part_dir)
        # Mass and Centre can be recovered
        kw_MCntr      = get_kwMCntr(Gn,SGn,z=z,sim=sim,gal_dir=gal_dir)
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
        mkdir(self.gal_snap_dir)
        #self.islens_file = self.gal_snap_dir/f"{self.Name}_islens.dll"
        
        self.run(reload=reload)

    ### Class Structure ####
    ########################
    def _identity(self):
        # Returns tuple to identify uniquely this galaxy
        Id = (self.sim,self.snap,self.Name)
        return Id
    
    def __hash__(self):
        # simplify the hash method
        return hash(self._identity())

    def __eq__(self, other):
        if not isinstance(other, PartGal):
            return NotImplemented
        return self._identity() == other._identity()
        
    def __str__(self):
        str_gal = f"Gal {self.Gn}.{self.SGn}"
        str_gal += f", at z={str(np.round(self.z,3))}/snap={self.snap},"
        str_gal += f" with \nN={'%.1E'%Decimal(self.N_tot_part)} part.\nof \ntot Mass={'%.1E'%Decimal(self.M_tot)} [M_sun]\n"
        str_gal +=f" divided in N \n\
                Stars:{'%.1E'%Decimal(self.N_stars)}\n\
                Gas:{'%.1E'%Decimal(self.N_gas)}\n\
                DM:{'%.1E'%Decimal(self.N_dm)}\n\
                BH:{'%.1E'%Decimal(self.N_bh)}\n"
        str_gal +=f"and Mass in \n\
                    Stars:{'%.1E'%Decimal(self.M_stars)} [M_sun]\n\
                    Gas:{'%.1E'%Decimal(self.M_gas)} [M_sun]\n\
                    DM:{'%.1E'%Decimal(self.M_dm)} [M_sun]\n\
                    BH:{'%.1E'%Decimal(self.M_bh)} [M_sun]\n"
        return str_gal 
    ########################
    ########################
    
    @property 
    def cosmo(self):
        return FlatLambdaCDM(H0=self.h*100, Om0=1-self.h)
    

    """
    def update_is_lens(self,islens,message="",kw_islens={}):
        kw_islens["islens"]  = islens
        kw_islens["message"] = message
        with open(self.islens_file,"wb") as f:
            dill.dump(kw_islens,f)
        return 0
    """
    def run(self,reload=True):
        upload_successful = False
        if reload:
            upload_successful = self.upload_prev(reload=reload)
        if not upload_successful:
            # useful to check Center of Mass
            self.xy_propr2comov = self.prop2comov("Coordinates") 
            self.m_propr2comov  = self.prop2comov("Mass") 
            self._init_path_snap()
            self._initialise_parts()
            # works:
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
        return read_snap_header(z=self.z,snap=self.snap,sim=self.sim,part_path=self.part_dir)
        
    def _init_path_snap(self):
        z_str          = prepend_str(str(int(self.z)),ln_str=3,fill="0")
        str_snap       = get_snap(self.z,3)
        fullsnap_str   = f"_{str_snap}_z{z_str}p???"
        self.path_snap = f"{self.part_dir}/{self.sim}/snapshot{fullsnap_str}/snap{fullsnap_str}"
        return 0
        
    def _initialise_parts(self):
        if not hasattr(self,"path_snap"):
            self._init_path_snap()
        self.gas   = self.read_part(0)
        self.dm    = self.read_part(1)
        self.stars = self.read_part(4)
        self.bh    = self.read_part(5)
        return 0
    
    def read_part(self,itype):
        """Read the particles properties and returns them in physical units
        """
        kw   = {}
        atts = ["GroupNumber","SubGroupNumber","Coordinates"]
        if itype!=1:
            atts.append("Mass")
            atts.append("SmoothingLength")
        else:
            """Special case for the mass of dark matter particles."""   
            fl =  glob.glob(f"{self.path_snap}.0.hdf5")
            if len(fl)!=1:
                raise RuntimeError(f"{fl} found to be not of lenght 1 from glob({self.path_snap}.0.hdf5): len={len(fl)}")
            fl = fl[0]
            with h5py.File(fl, 'r') as f:
                h = f['Header'].attrs.get('HubbleParam')
                a = f['Header'].attrs.get('Time')
                dm_mass = f['Header'].attrs.get('MassTable')[1]
                n_particles = f['Header'].attrs.get('NumPart_Total')[1]
                # Create an array of lenght n_particles each set to dm_mass
                m = np.ones(n_particles, dtype='f8') *dm_mass
                # Use the conversion factors from the mass entry in the gas particles.
                cgs  = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
                aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
                hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
            # Convert to proper/physical mass 
            kw["Mass"] = np.multiply(m, cgs*(a**aexp)*(h**hexp), dtype='f8')
            
        nfiles = 16
        files = glob.glob(f"{self.path_snap}.{[i for i in range(nfiles)]}.hdf5".replace(" ",""))
        if len(files)!=nfiles:
                raise RuntimeError(f"Found {len(files)} files instead of {nfiles}")
        for att in atts:
            data = []
            for i in range(nfiles):
                fl =  files[i]
                with h5py.File(fl,'r') as f:
                    tmp = f['PartType%i/%s'%(itype,att)][...]
                    data.append(tmp)
                    # Get conversion factors
                    cgs  = f['PartType%i/%s'%(itype,att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype,att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype,att)].attrs.get('h-scale-exponent')
                    # Get expansion factor and Hubble parameter from the header
                    a = f['Header'].attrs.get('Time')
                    h = f['Header'].attrs.get('HubbleParam')
            if len(tmp.shape) > 1:
                data = np.vstack(data)
            else:
                data = np.concatenate(data)
            # Convert to physical/proper
            if data.dtype!=np.int32 and data.dtype!=np.int64:
                # note: it IS multiply by 1/h (h**hexp)
                data = np.multiply(data,cgs*(a**aexp)*(h**hexp),dtype='f8')
            kw[att] = data
        #print("self.Gn,self.SGn",self.Gn,self.SGn)
        mask   = np.logical_and(kw["GroupNumber"]==self.Gn,kw["SubGroupNumber"]==self.SGn)
        output         = {}
        output["mass"] = kw["Mass"][mask] * u.g.to(u.Msun)
        coords         = kw["Coordinates"][mask]*u.cm.to(u.Mpc)
        # Periodic wrap coordinates around centre.
        boxsize        = self.boxsize*(h**hexp)
        centre         = self.centre*(a**aexp) # given in comoving (but not 1/h)
        output['coords'] = np.mod(coords-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize
        if itype!=1:
            output["smooth"] = kw["SmoothingLength"][mask]*u.cm.to(u.Mpc)
        return output
        
    def _count_tot_part(self):
        """Count total particle number"""
        self.N_gas      = _count_part(self.gas)
        self.N_dm       = _count_part(self.dm)
        self.N_stars    = _count_part(self.stars)
        self.N_bh       = _count_part(self.bh)
        self.N_tot_part =  self.N_gas+self.N_dm+self.N_stars+self.N_bh
        return self.N_tot_part

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
        
    @property
    def gal_snap_dir(self):
        """Define/locate main snap/redshift directory
        """
        gal_snap_dir = self.gal_dir/f"snap_{self.snap}"
        return gal_snap_dir
    @property
    def pkl_path(self):
        """Define pkl path to store the class instance
        """
        pkl_path = self.gal_snap_dir/f"{self.Name}.pkl"
        return pkl_path
        
    def store_gal(self):
        # store this galaxy
        with open(self.pkl_path,"wb") as f:
            dill.dump(self,f)
        print(f"Saved {self.pkl_path}")

# this function is a wrapper for convenience - it takes the class itself as input
def ReadGal(Gal,vebose=True):
    return LoadClass(path=Gal.pkl_path,verbose=verbose)

def LoadGal(path,if_fail_recompute=True,sim=std_sim,verbose=True):
    # Try loading galaxy - if fail and fail_recompute==True, try recomputing it
    Gal = LoadClass(path=path,verbose=verbose)
    if not Gal and if_fail_recompute:
        gal_dir = sim2galdir(sim)
        kwGal   = gal_path2kwGal(path,gal_dir=gal_dir)
        Gal     = PartGal(**kwGal)
    return Gal
    
# to simplify the input: given the sim, z, and GnSgn, 
# we get the mass and center of the galaxy for input of PartGal 


def get_myCat(Gn,SGn,z,sim,min_mass="1e10",dz=0.05,gal_dir=std_gal_dir):
    min_z=str(z-0.05)
    max_z=str(z+0.05)

    # try first finding the exact one
    cat_path = get_catpath(min_mass=min_mass,\
                           min_z=min_z,max_z=max_z,
                          gal_dir=gal_dir)
    if os.path.isfile(cat_path):
        myCat = load_whatever(cat_path)
    else:
        # try all the other
        found = False
        for pkl in Path(gal_dir).glob("*.pkl"):
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
                z=None,snap=None,
               gal_dir=std_gal_dir):
    z,snap       = get_z_snap(z,snap)
    myCat,index  = get_myCat(Gn,SGn,z,sim,gal_dir=gal_dir)
    Centre       = np.array([myCat["CMx"],myCat["CMy"],myCat["CMz"]]).T[index]
    kwMCntr      = {"M":myCat["M"][index],
                    "Centre":Centre}
    return kwMCntr

def get_rnd_PG(sim=std_sim,min_mass = min_mass,min_z=min_z,max_z=max_z,
               check_prev=True,save_pkl=True):
    """Randomly sample a galaxy from the simulation 
    """

    kw_gal   = get_rnd_gal_indexes(sim=sim,min_mass=min_mass,min_z=min_z,max_z=max_z,
                                   check_prev=check_prev,save_pkl=save_pkl)
    z        = kw_gal["z"] 
    M        = kw_gal["M"] 
    Gn,SGn   = kw_gal["Gn"],kw_gal["SGn"]
    Centre   = np.array([kw_gal["CMx"],kw_gal["CMy"],kw_gal["CMz"]]) 
    kwGal    = {"z":z,"Gn":Gn,"SGn":SGn,"sim":sim,"M":M,"Centre":Centre}
    PG       = PartGal(**kwGal)
    return PG

def clip_coord(m,x,y,z,sigma=10):
    # clip coordinates outliers
    mask = np.ones(len(x),dtype=bool)
    for coord in x,y,z:
        mask *= np.invert(sigma_clip(coord,sigma=sigma).mask)
    return m[mask],x[mask],y[mask],z[mask]
    

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

def Gal2kwMXYZ(Gal): 
    Ms, Xs,Ys,Zs = Gal2MXYZ(Gal)
    return {"Ms":Ms,"Xs":Xs,"Ys":Ys,"Zs":Zs}
    
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
        
# for debug:
if __name__=="__main__":
    
    PG = get_rnd_PG()
    fig, ax = plt.subplots(3)
    nx = 100
    for name,part in zip(["stars","dm","gas"],[PG.stars,PG.dm,PG.gas]):
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
    """
    print("Testing dens map")
    from test_proj_part_hist import get_dens_map_rotate_hist
    NG.proj_dir = "./tmp/proj_part"
    mkdir(NG.proj_dir)
    dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=NG,pixel_num=100j,
                                                            z_source_max=5,verbose=True,plot=True)
    """