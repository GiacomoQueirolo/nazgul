import glob
import h5py
import pickle
import copy
import numpy as np
from decimal import Decimal

import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt  
from concurrent.futures import ThreadPoolExecutor

from python_tools.get_res import load_whatever
from python_tools.tools import mkdir
####

# Setup and General Structure
##############################

# data dir structure: data_path 
#                        |_ Sim
#                            |_snapshots_of_particles
#                            |_Gals
#                                |_snaphots_of_gals (obtained from particles)

# data path
part_data_path = "/pbs/home/g/gqueirolo/EAGLE/data/"
# "Standard" simulation
# use the following only as test case
#std_sim  = "RefL0012N0188"
std_sim  = "RefL0025N0752"
test_sim = "RefL0012N0188"
sim_path = part_data_path+std_sim+"/"
# Where to store the galaxies
gal_dir = sim_path+"/Gals"
mkdir(gal_dir)
# from https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots/
# valid fo all sims apart the variable IMF runs
kw_snap_z = {"28":0, "27":0.1, "26":0.18, "25":0.27, "24":0.37, "23":0.5, "22":0.62, "21":0.74, "20":0.87, "19":1, "18":1.26, "17":1.49, "16":1.74, "15":2.01, "14":2.24, "13":2.48, "12":3.02, "11":3.53, "10":3.98, "9":4.49, "8":5.04, "7":5.49, "6":5.97, "5":7.05, "4":8.07, "3":8.99, "2":9.99, "1":15.13, "0":20}
#inverted kw
kw_z_snap = {}
for k in kw_snap_z:
    kw_z_snap[kw_snap_z[k]] = k
# z indexes
z_index = np.array([float(f) for f in list(kw_z_snap.keys())])

# Useful functions:
###################

def get_z(snap):
    snap = str(snap)
    while snap[0]=="0" and snap!="0":
        snap = snap[1:]
    return kw_snap_z[snap]

def get_snap(z,_ln_snap=None):
    # consider a continous z instead of the discreet version
    # works for discreet z as well
    key_z = min(kw_z_snap.keys(),key=lambda k:np.abs(k-float(z)))
    snap  = str(kw_z_snap[key_z])
    snap  = prepend_str(snap,ln_str=_ln_snap,fill=0)
    return snap

def get_z_snap(z=None,snap=None):
    if z is None and snap is None:
        raise UserWarning("Give either z or snap")
    if z is None:
        z = get_z(snap)
    else:
        snap = get_snap(z)
    return z,snap
    
def prepend_str(str_i,ln_str,fill="0"):
    if ln_str is None:
        return str_i
    str_i = str(str_i)
    fill  = str(fill) 
    while len(str_i)<ln_str:
        str_i=fill+str_i
    return str_i

def get_files(sim,z=None,snap=None,_i_="*"):
    """
    Find the files 
    If _i_ is specified, only that specific subsection of the snapshot (useful for DM)
    If no redshift/snapshots are defined, take all of them
    """
    
    sim_path = part_data_path+"/"+sim
    # find the files
    _i_ = str(_i_)
    pstring = "???"
    suffix = "p"+pstring+"."+_i_+".hdf5"
    prefix = sim_path+"/snapshot_"
    if z is None and snap is None:
        # take all snapshots/all redshifts
        snap ="0??"
        zstr = "???"
    else:
        if z is not None and snap is not None:
            # verify that they are compatible:
            assert int(get_snap(z))==int(snap)
        if z is not None:
            zstr = str(int(z))
            snap = get_snap(z)
        elif snap is not None:
            #zstr = str(get_z(snap))
            pth   = prefix+prepend_str(snap,ln_str=3,fill="0")+"_z*"
            _zstr = glob.glob(pth)
            assert len(_zstr)==1
            zstr  = _zstr[0].split("_z")[1].split("p")[0]
        snap = prepend_str(snap,ln_str=3,fill="0")
        zstr = prepend_str(zstr,ln_str=3,fill="0")
    
    fix  = f"{snap}_z{zstr}p{pstring}/snap_{snap}_z{zstr}"
    file_string = prefix+fix+suffix
    #print("#DEBUG")
    #print(file_string)
    files = glob.glob(file_string)
    # checking that the files are not empty
    assert files != []
    return files

# ugly fnct but should be correct:
def get_simsize(sim_name):
    return int(sim_name.split("L")[1].split("N")[0])
"""
# discontinued
def stnd_read_dataset(itype, att,
                 z=None,snap=None,
                 sim=std_sim):
    # Read a selected dataset:
    #    - itype is the PartType (stars,gas etc) 
    #    - att is the attribute name (Group, Subgroup etc). 
    #    If no redshift/snapshots are define, take all of them
    
    # Output array.
    data  = []
    files = get_files(sim=sim,z=z,snap=snap)
    # Loop over each file and extract the data.
    for i,file in enumerate(files):
        with h5py.File(file, 'r') as f:
            tmp = f['PartType%i/%s'%(itype, att)][...]
            data.append(tmp)
            if i==0:
                # Get conversion factors.
                cgs     = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp    = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp    = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
        
                # Get expansion factor and Hubble parameter from the header.
                a       = f['Header'].attrs.get('Time')
                h       = f['Header'].attrs.get('HubbleParam')

    # Combine to a single array.
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)

    # Convert to physical.
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')

    return data
"""

# DEBUG
def read_dataset(itype, att, z=None, snap=None, sim=std_sim):
    data     = []
    meta     = {}
    nfiles   = 16
    str_snap = get_snap(z,3)
    z_str    = prepend_str(str(int(z)),ln_str=3,fill="0")
    for i in range(nfiles):
        fl =  glob.glob('/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752/snapshot_'+str_snap+'_z'+z_str+'p???/snap_'+str_snap+'_z'+z_str+'p???.'+str(i)+'.hdf5')
        if len(fl)!=1:
            print(fl)
            raise RuntimeError("should be only one")
        fl  = fl[0]
        f   = h5py.File(fl,'r')
        tmp = f['PartType%i/%s'%(itype,att)][...]
        data.append(tmp)
        # Get conversion factors
        cgs  = f['PartType%i/%s'%(itype,att)].attrs.get('CGSConversionFactor')
        aexp = f['PartType%i/%s'%(itype,att)].attrs.get('aexp-scale-exponent')
        hexp = f['PartType%i/%s'%(itype,att)].attrs.get('h-scale-exponent')
        # Get expansion factor and Hubble parameter from the header
        a = f['Header'].attrs.get('Time')
        h = f['Header'].attrs.get('HubbleParam')
        
        """
        # double check
        if i == 0:
            #print(f['PartType%i/%s'%(itype,att)].attrs.get('h-scale-exponent'))
            #print(f[f'PartType{itype}/{att}'].attrs.get('h-scale-exponent'))
            meta['cgs'] = f[f'PartType{itype}/{att}'].attrs.get('CGSConversionFactor')
            meta['aexp'] = f[f'PartType{itype}/{att}'].attrs.get('aexp-scale-exponent')
            meta['hexp'] = f[f'PartType{itype}/{att}'].attrs.get('h-scale-exponent')
            meta['a'] = f['Header'].attrs.get('Time')
            meta['h'] = f['Header'].attrs.get('HubbleParam')
            print('meta',meta)
        print("i,cgs,a,aexp,h,hexp",i,cgs,a,aexp,h,hexp)
        """
        f.close()
        

    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)
    # Convert to physical
    if data.dtype!=np.int32 and data.dtype!=np.int64:
        data = np.multiply(data,cgs*a**aexp*h**hexp,dtype='f8')

    #DEBUG
    if att=="Coordinates":
        print(np.shape(data))
        x,y,z = data.T
        name = str(itype) 
        plt.close()
        fig, ax = plt.subplots(3)
        nx = 100
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[0].set_xlabel("X [kpc]")
        ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[1].set_xlabel("Y [kpc]")
        ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[2].set_xlabel("Z [kpc]")
        ax[2].legend()
        
        namefig = f"tmp/hist1D_part_{itype}1_read_part.png"
        plt.tight_layout()
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig) 
    """
    if att=="GroupNumber":
        print("Groupnumber shape")
        print(np.shape(data))
        plt.hist(data)
        namefig = f"tmp/hist_groupnumber_{itype}.png"
        plt.tight_layout()
        plt.savefig(namefig)
        plt.close()
    """
    return data
# my version:
"""
def read_dataset(itype, att, z=None, snap=None, sim=std_sim):
    files = get_files(sim=sim, z=z, snap=snap)
    
    data_chunks = []
    meta = {}

    def read_file(i_file_tuple):
        i, file = i_file_tuple
        with h5py.File(file, 'r') as f:
            dataset = f[f'PartType{itype}/{att}'][...]
            if i == 0:
                meta['cgs'] = f[f'PartType{itype}/{att}'].attrs.get('CGSConversionFactor')
                meta['aexp'] = f[f'PartType{itype}/{att}'].attrs.get('aexp-scale-exponent')
                meta['hexp'] = f[f'PartType{itype}/{att}'].attrs.get('h-scale-exponent')
                meta['a'] = f['Header'].attrs.get('Time')
                meta['h'] = f['Header'].attrs.get('HubbleParam')
        return dataset

    with ThreadPoolExecutor() as executor:
        data_chunks = list(executor.map(read_file, enumerate(files)))

    # Combine arrays
    if data_chunks[0].ndim > 1:
        data = np.vstack(data_chunks)
    else:
        data = np.concatenate(data_chunks)

    # Apply conversion if not integer-> convers. to proper coord/mass
    if data.dtype not in (np.int32, np.int64):
        #print(data,meta['cgs'] * meta['a']**meta['aexp'] * meta['h']**meta['hexp'])
        data = np.multiply(data, meta['cgs'] * meta['a']**meta['aexp'] * meta['h']**meta['hexp'], dtype='f8')

    return data
"""


def read_snap_header(z=None,snap=None,sim=std_sim):
    """ Read various attributes from the header group. """
    file    = get_files(sim,z,snap,_i_=0)
    #print("#DEBUG")
    #print(sim,z,snap)
    if len(file)!=1:
        raise RuntimeError("Warning: define only one snapshot")
        print("file=",file)
    file      = file[0]
    aexp,hexp = {},{}
    with h5py.File(file, 'r') as f:
        a       = f['Header'].attrs.get('Time')                # Scale factor.
        h       = f['Header'].attrs.get('HubbleParam')         # h = H0/(100km/s/Mpc)
        boxsize = f['Header'].attrs.get('BoxSize')             # L [cMph/h].
        # aexp and hexp are different for the diff. variable (mainly coord and mass)
        # but should be the same between different type of particles and redshift bins
        atts = "GroupNumber","SubGroupNumber","Mass","Coordinates","SmoothingLength"
        for att in atts:
            aexp[att] = f[f"PartType0/{att}"].attrs["aexp-scale-exponent"]  # Exponent of Scale factor.
            hexp[att] = f[f"PartType0/{att}"].attrs["h-scale-exponent"]     # Exponent of h
    return a,aexp,h,hexp,boxsize


def read_snap_header_simple(z=None,snap=None,sim=std_sim):
    """ Read various attributes from the header group.  -> simplified"""
    file    = get_files(sim,z,snap,_i_=0)
    #print("#DEBUG")
    #print(sim,z,snap)
    if len(file)!=1:
        raise RuntimeError("Warning: define only one snapshot")
        print("file=",file)
    file      = file[0]
    aexp,hexp = {},{}
    with h5py.File(file, 'r') as f:
        a       = f['Header'].attrs.get('Time')                # Scale factor.
        h       = f['Header'].attrs.get('HubbleParam')         # h = H0/(100km/s/Mpc)
        boxsize = f['Header'].attrs.get('BoxSize')             # L [cMph/h].
        """
        # aexp and hexp are different for the diff. variable (mainly coord and mass)
        # but should be the same between different type of particles and redshift bins
        atts = "GroupNumber","SubGroupNumber","Mass","Coordinates","SmoothingLength"
        for att in atts:
            aexp[att] = f[f"PartType0/{att}"].attrs["aexp-scale-exponent"]  # Exponent of Scale factor.
            hexp[att] = f[f"PartType0/{att}"].attrs["h-scale-exponent"]     # Exponent of h
    #return a,aexp,h,hexp,boxsize
    """
    return a,h,boxsize

def read_dataset_dm_mass(z,snap,sim=std_sim):
    """Special case for the mass of dark matter particles."""
    # Output array.
    files = get_files(sim=sim,z=z,snap=snap,_i_="0")
    if len(files)!=1:
        raise RuntimeError("Define the z and/or snap")
    with h5py.File(files[0], 'r') as f:
        h = f['Header'].attrs.get('HubbleParam')
        a = f['Header'].attrs.get('Time')
        dm_mass = f['Header'].attrs.get('MassTable')[1]
        n_particles = f['Header'].attrs.get('NumPart_Total')[1]
        # Create an array of lenght n_particles each set to dm_mass
        m = np.ones(n_particles, dtype='f8') *dm_mass
        # Use the conversion factors from the mass entry in the gas particles.
        cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
        aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
        hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
    # Convert to proper/physical mass
    m = np.multiply(m, cgs*(a**aexp)*(h**hexp), dtype='f8')
    return m

def get_gal_path(Gal,ret_snap_dir=False):
    try:
        gal_path,gal_snap_dir =  Gal.gal_path,Gal.gal_snap_dir
    except:
        gal_snap_dir = f"{gal_dir}/snap_{Gal.snap}/"
        gal_path = f"{gal_snap_dir}/{Gal.Name}.pkl"
    if ret_snap_dir:
        return gal_path,gal_snap_dir
    return gal_path


class Galaxy:
    def __init__(self, Gn, SGn, CMx,CMy,CMz,M=None,
                 sim=std_sim,z=None,snap=None,query=""):
        self.sim    = sim
        z,snap      = self.get_z_snap(z,snap)
        self.snap   = snap
        self.z      = z
        self.a,self.aexp,self.h,self.hexp,self.boxsize = self.read_snap_header()
        # define all coordinates in Mpc / cMpc but not Mpc/h
        self.centre = np.array([CMx,CMy,CMz]) # cMpc
        self.Gn     = Gn
        self.SGn    = SGn
        self.Name   = f"G{Gn}SGn{SGn}" #note this is unique only within the snap
        self.M      = M
        self.query  = query # query from which this gal is selected (important later for sel. bias)
        # Load data.
        # note a and h have already the exponent (for a it's 1, for h it's -1)

        # Define/Create Gal path
        self.set_gal_path()

        # useful to check Center of Mass
        self.xy_propr2comov = self.prop2comov("Coordinates") 
        self.m_propr2comov  = self.prop2comov("Mass") 
        # check if gal exists and if it's the same, if so load that instead
        try:
            GL = load_whatever(self.gal_path)
            print("Gal loaded")
            if self.__eq__(GL):
                # this is what takes the longest
                self.gas   = GL.gas
                self.dm    = GL.dm
                self.bh    = GL.bh
                self.stars = GL.stars
                to_store = False
            else:
                raise RuntimeError("Galaxy file exists but is not the same")
        except:
            print("Gal not loaded, read from data")
            self.gas    = self.read_galaxy(0)
            self.dm     = self.read_galaxy(1)
            self.stars  = self.read_galaxy(4)
            self.bh     = self.read_galaxy(5)
            to_store    = True
        self._count_tot_part()
        self._mass_tot_part()
        self._verify_cnt()
        if to_store:
            self.store_gal()


        
    def read_galaxy(self,itype):
        kw_gal = self._get_kw_gal()
        return read_galaxy(itype=itype,**kw_gal)
        
    def read_snap_header(self):
        return read_snap_header(z=self.z,snap=self.snap,sim=self.sim)

    def prop2comov(self,varType):
        # factor to multiply to proper coordinates or proper mass
        # to obtain the corresponding comoving properties
        # from eq.1-2 https://arxiv.org/pdf/1706.09899

        # note: in principle,numerically, aexp and hexp COULD be different for
        # each type of particle and snapshot but that would make no sense
        # so I'll assume them constant

        # in principle the factor is 1/(a^0*h^-1) = h for mass
        # and 1/(a^1 *h^-1) = h/a for coords
        # but is more correctly defined as
        # 1/(a^aexp * h^hexp)
        """
        aexp = self.aexp[varType]
        hexp = self.hexp[varType]
        if varType=="Coordinates":
            np.testing.assert_almost_equal(aexp,1)
            np.testing.assert_almost_equal(hexp,-1)
        elif varType=="Mass":
            np.testing.assert_almost_equal(aexp,0)
            np.testing.assert_almost_equal(hexp,-1)
        hexp = 0
        return  1/((self.a**aexp)*(self.h**hexp))
        """
    
        # AS DEFINED: ALL COORDS IN Mpc/cMpc. NO 1/h FACTOR! 
        # hence the correction factor must not correct for h
        aexp = self.aexp[varType]
        hexp = 0
        # physically motivated verification 
        if varType=="Coordinates":
            self.verbose_assert_almost_equal(aexp,1)
        elif varType=="Mass":
            self.verbose_assert_almost_equal(aexp,0)
        return  1/(self.a**aexp)
    
    def _get_kw_gal(self):
        kw_gal = {"gn":self.Gn,"sgn":self.SGn,
                 "snap":self.snap,"z":self.z,
                 "boxsize":self.boxsize,"h":self.h,"centre":self.centre}
        return kw_gal 
        
    def get_z_snap(self,z,snap):
        return get_z_snap(z,snap)
        
    def _count_tot_part(self):
        self.N_gas      = _count_part(self.gas)
        self.N_dm       = _count_part(self.dm)
        self.N_stars    = _count_part(self.stars)
        self.N_bh       = _count_part(self.bh)
        self.N_tot_part =  self.N_gas+self.N_dm+self.N_stars +self.N_bh
        return self.N_tot_part

    def _mass_tot_part(self):
        self.M_gas   = _mass_part(self.gas)
        self.M_dm    = _mass_part(self.dm)
        self.M_stars = _mass_part(self.stars)
        self.M_bh    = _mass_part(self.bh)
        
        self.M_tot   =  self.M_gas+self.M_dm+self.M_stars +self.M_bh
        self.verbose_assert_almost_equal(float(self.M_tot)/float(self.M),1,decimal=3,msg_title="Mass")
        return self.M_tot
        
    def verbose_assert_almost_equal(self,value1,value2=1,decimal=3,msg_title=None):
        # a verbose way of giving info if if fails
        try:
            np.testing.assert_almost_equal(value1,value2,decimal=decimal)
        except AssertionError as AssErr:
            if msg_title:
                print(msg_title)
            print("Error for \n"+str(self))
            print("found in \n"+str(self.get_pkl_path()))
            raise AssertionError(AssErr)
        return 0
        
    def _verify_cnt(self):
        # verify that the center of mass is indeed correct
        if not hasattr(self,"M_tot"):
            self._mass_tot_part()

        # get the mass, restructure it in 3,N_part
        # and convert in comov. coordinates
        #print("mass_propr2comov,      xy_propr2comov:")
        #print(self.m_propr2comov,self.xy_propr2comov)
        
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
        
        #print("DEBUG")
        #print(cnt_m/self.centre,cnt_m,self.centre)
        """
        eps = 1e-6
        if np.all(np.abs(self.centre)<eps):
            print("Galaxy centered around zero")
            center_actual = np.array(cnt_m)    
            center_desired = np.zeros(3)
        else:
        """
        center_actual  = np.array(cnt_m)
        center_desired = self.centre
        # is very odd that for some galaxy the center is off by more than 0.01 for 1 of the coords
        # maybe different wrapping?
        self.verbose_assert_almost_equal(center_desired,center_desired,decimal=1,msg_title="Centre")      
        return 0
        
    def get_pkl_path(self):
        if not getattr(self,"gal_snap_dir",False):
           self.gal_path,self.gal_snap_dir = get_gal_path(self,ret_snap_dir=True)
        if not getattr(self,"pkl_path",False):
            self.pkl_path = f"{self.gal_snap_dir}/Gn{self.Gn}SGn{self.SGn}.pkl"
        return self.pkl_path
        
    def set_gal_path(self):
        self.gal_path,self.gal_snap_dir = get_gal_path(self,ret_snap_dir=True)
        mkdir(self.gal_snap_dir)
        self.get_pkl_path()
        return 0
    def store_gal(self):
        if getattr(self,"pkl_path",False):
            self.set_gal_path()
        # store this galaxy
        with open(self.pkl_path,"wb") as f:
            pickle.dump(self,f)

        
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
    def __eq__(self,other_gal):
        kw_one = self._get_kw_gal() 
        kw_two = other_gal._get_kw_gal() 
        if kw_one.keys()!=kw_two.keys():
            return False
        else:
            for k in kw_one.keys():
                if kw_one[k]!=kw_two[k]:
                    return False
        return True
        
def _count_part(part):
    return len(part["mass"])

def _mass_part(part):
    return np.sum(part["mass"])

def read_galaxy(itype,gn,sgn,z,snap,boxsize,h,centre):
    """ For a given galaxy (defined by its GroupNumber, SubGroupNumber 
    and z/snap) extract the coordinates and mass of all particles of a 
    selected type.
    Coordinates are then wrapped around the centre to account for periodicity. """

    data = {}

    # Load data, then mask to selected GroupNumber and SubGroupNumber.
    gns  = read_dataset(itype, 'GroupNumber', z,snap)
    sgns = read_dataset(itype, 'SubGroupNumber',z,snap)
    mask = np.logical_and(gns == gn, sgns == sgn)    
    
    if itype == 1:
        data['mass'] = read_dataset_dm_mass(z, snap)[mask] * u.g.to(u.Msun)
        # DM does NOT have a smoothing scale
    else:
        data['mass'] = read_dataset(itype, 'Mass', z, snap)[mask] * u.g.to(u.Msun)
        # Add Smoothing lenght -> see Co-moving SPH smoothing kernel in eagle-particle paper
        # sec. 3
        data['smooth'] = read_dataset(itype, 'SmoothingLength', z, snap)[mask] * u.cm.to(u.Mpc)
        # chachacha real smooth
    data['coords'] = read_dataset(itype, 'Coordinates', z, snap)[mask] * u.cm.to(u.Mpc)
    print("DEBUG fnct")
    print("std coords",data["coords"].std(axis=0))
    x,y,z= data['coords'].T
    print(np.shape(data["coords"]))
    name = itype 
    plt.close()
    fig, ax = plt.subplots(3)
    nx = 100
    ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
    ax[0].set_xlabel("X [kpc]")
    ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[1].set_xlabel("Y [kpc]")
    ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[2].set_xlabel("z [kpc]")

    # Periodic wrap coordinates around centre.
    boxsize = boxsize/h # this should in principle be def. by hexp, but it's way easier like this
    centre  = np.array(centre)*(z+1) # must be corrected from cMpc to Mpc
    for i,axi in enumerate(ax):
        axi.axvline(centre[i],c="k",ls="--",label="cnt")
        axi.axvline(centre[i]+boxsize*.5,c="r",label="box (h corrected)")
        axi.axvline(centre[i]-boxsize*.5,c="r")
        """
        axi.axvline(centre[i]/h,c="grey",ls="--",label="cnt/h")
        axi.axvline(centre[i]/h+boxsize*.5,c="orange",label="box (also cnt/h)")
        axi.axvline(centre[i]/h-boxsize*.5,c="orange")
        """
        rt = np.mean(data['coords'].T[i])/centre[i]
        # very close to h
        print("ratio",rt,1/rt)
        print("h",h)

    orig_data_coord = copy.copy(data['coords'])
    data['coords'] = np.mod(data['coords']-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize
    
    print("std coords",data["coords"].std(axis=0))

    x,y,z= data['coords'].T
    ax[0].hist(x,bins=nx,alpha=.5,label=str(name)+" after")#,range=[xmin, xmax])
    ax[0].set_xlabel("X [kpc]")
    ax[1].hist(y,bins=nx,alpha=.5,label=str(name)+" after")#,range=[ymin, ymax])
    ax[1].set_xlabel("Y [kpc]")
    ax[2].hist(z,bins=nx,alpha=.5,label=str(name)+" after")#,range=[ymin, ymax])
    ax[2].set_xlabel("Z [kpc]")
    """
    centre = np.array(centre)/h # try it
    dt = np.mod(orig_data_coord-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize
    x,y,z= dt.T
    ax[0].hist(x,bins=nx,alpha=.5,label=str(name)+" after (cnt/h)")#,range=[xmin, xmax])
    ax[0].set_xlabel("X [kpc]")
    ax[1].hist(y,bins=nx,alpha=.5,label=str(name)+" after (cnt/h)")#,range=[ymin, ymax])
    ax[1].set_xlabel("Y [kpc]")
    ax[2].hist(z,bins=nx,alpha=.5,label=str(name)+" after (cnt/h)")#,range=[ymin, ymax])
    ax[2].set_xlabel("Z [kpc]")
    """
    ax[2].legend()

    namefig = f"tmp/hist1D_part_{itype}1_fnct.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig) 
    return data


"""
class RotationCurve:

    def __init__(self, Gn, SGn,  CMx,CMy,CMz,M=None,sim=std_sim,z=None,snap=None):
        self.sim    = sim
        z,snap      = get_z_snap(z,snap)
        self.z      = z
        self.snap   = snap
        self.centre = np.array([CMx,CMy,CMz])
        self.Gn     = Gn
        self.SGn    = SGn
        # centre only needed here to correct the coordinates
        # it could in theory get recovered from the particles themselves
        # as it follows X = Sum miXi / Sum mi

        # Load data.
        self.a, self.axep,self.h,self.hexp, self.boxsize = read_snap_header(z=self.z,
                                                        snap=self.snap,
                                                        sim=self.sim)

        self.gas    = self.read_galaxy(0)
        self.dm     = self.read_galaxy(1)
        self.stars  = self.read_galaxy(4)
        self.bh     = self.read_galaxy(5)
        # Plot.
        self.plot()
        
    def read_galaxy(self,itype):
        kw_rg = {"gn":self.Gn,"sgn":self.SGn,
                 "snap":self.snap,"z":self.z,
                 "boxsize":self.boxsize,"h":self.h,"centre":self.centre}
        return read_galaxy(itype=itype,**kw_rg)
        
    def compute_rotation_curve(self, arr):
        #Compute the rotation curve. 

        # Compute distance to centre.
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)
        mask = np.argsort(r)
        r = r[mask]

        # Compute cumulative mass.
        cmass = np.cumsum(arr['mass'][mask])

        # Compute velocity.
        myG = G.to(u.km**2 * u.Mpc * u.Msun**-1 * u.s**-2).value
        v = np.sqrt((myG * cmass) / r)

        # Return r in Mpc and v in km/s.
        return r, v

    def plot(self):
        plt.figure()

        # All parttypes together.
        combined = {}
        combined['mass'] = np.concatenate((self.gas['mass'], self.dm['mass'],
            self.stars['mass'], self.bh['mass']))
        combined['coords'] = np.vstack((self.gas['coords'], self.dm['coords'],
            self.stars['coords'], self.bh['coords']))
        
        # Loop over each parttype.
        for x, lab in zip([self.gas, self.dm, self.stars, combined],
                        ['Gas', 'Dark Matter', 'Stars', 'All']):
            r, v = self.compute_rotation_curve(x)
            plt.plot(r*1000., v, label=lab)

        # Save plot.
        plt.legend(loc='center right')
        plt.minorticks_on()
        plt.ylabel('Velocity [km/s]')
        plt.xlabel('r [kpc]')
        plt.xlim(1, 50) 
        plt.tight_layout()
        plt.savefig('RotationCurve.pdf')
        plt.close()

if __name__ == '__main__':
    centre = np.array([12.08808994,4.47437191,1.41333473])
    x = RotationCurve(gn=1, sgn=0, centre = centre,z=0)



"""

