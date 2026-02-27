"""
Random bazar of useful functions for the reading of particles for galaxies
"""
import glob
import h5py
import pickle
import numpy as np
# Implement path handling 
from pathlib import Path

from python_tools.get_res import load_whatever
from python_tools.tools import mkdir,to_dimless
####

# Setup and General Structure
##############################

# data dir structure: data_path 
#                        |_ Sim
#                            |_snapshots_of_particles
#                            |_Gals
#                                |_snaphots_of_gals (obtained from particles)


# data path
#part_data_path = Path("/pbs/home/g/gqueirolo/EAGLE/data/")
part_data_path = Path("./data/")
# "Standard" simulation
std_sim  = "RefL0025N0752"
# use the following simulation only as test case
test_sim = "RefL0012N0188"
# used for tutorial -linked ONLY snap 20 of test_sim
tutorial_sim = "RefTuto"

def galdir2sim(gal_dir):
    sim_path = gal_dir.parent
    sim      = str(sim_path.name)
    return sim

def sim2galdir(sim,part_path=part_data_path):
    sim_path = Path(part_path)/sim
    gal_dir  = sim_path/"Gals"
    mkdir(gal_dir)
    return gal_dir

# Where to store the galaxies
std_gal_dir = sim2galdir(std_sim,part_path=part_data_path) 

# ./tmp will be a collector of intermediate, mildly useful plots/results, with the advantage of being easily accessible
mkdir("./tmp")

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
    # strip all leading zeros
    snap = str(snap).lstrip("0")
    if snap=="":
        snap="0"
    return kw_snap_z[snap]

def get_snap(z,_ln_snap=None):
    # consider a continous z instead of the discreet version
    # works for discreet z as well
    key_z = min(kw_z_snap.keys(),key=lambda k:np.abs(k-float(z)))
    snap  = str(kw_z_snap[key_z])
    snap  = prepend_str(snap,ln_str=_ln_snap,fill=0)
    return snap

def get_z_snap(z=None,snap=None):
    """Given either z or snap, return both
    """
    if z is None and snap is None:
        raise UserWarning("Give either z or snap")
    if z is None:
        z = get_z(snap)
    else:
        snap = get_snap(z)
    snap = str(snap)
    return z,snap
    
def prepend_str(str_i,ln_str,fill="0"):
    """Prepend 'fill' to 'str_i' until it reaches the lenght 'ln_str'
    """
    if ln_str is None:
        return str_i
    str_i = str(str_i)
    fill  = str(fill) 
    while len(str_i)<ln_str:
        str_i=fill+str_i
    return str_i

def get_files(sim,z=None,snap=None,_i_="*",part_path=part_data_path):
    """
    Find the files 
    If _i_ is specified, only that specific subsection of the snapshot (useful for DM)
    If no redshift/snapshots are defined, take all of them
    """
    
    sim_path = Path(part_path)/sim
    # find the files
    _i_ = str(_i_)
    pstring = "???"
    suffix = "p"+pstring+"."+_i_+".hdf5"
    prefix = sim_path/"snapshot_"
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
            _zstr = glob.glob(str(prefix)+prepend_str(snap,ln_str=3,fill="0")+"_z*")
            assert len(_zstr)==1
            zstr  = _zstr[0].split("_z")[1].split("p")[0]
        snap = prepend_str(snap,ln_str=3,fill="0")
        zstr = prepend_str(zstr,ln_str=3,fill="0")
    
    fix  = f"{snap}_z{zstr}p{pstring}/snap_{snap}_z{zstr}"
    file_string = f"{prefix}{fix}{suffix}"
    files = glob.glob(file_string)
    # checking that the files are not empty
    if files == []:
        raise RuntimeError(f"Files not found:{file_string}")
    return files


def read_snap_header(z=None,snap=None,sim=std_sim,part_path=part_data_path):
    """Read various attributes from the header group. 
    """
    file    = get_files(sim,z,snap,_i_=0,part_path=part_path)
    if len(file)!=1:
        print("file=",file)
        raise RuntimeError("Warning: define only one snapshot")
    file      = file[0]
    with h5py.File(file, 'r') as f:
        a       = f['Header'].attrs.get('Time')                # Scale factor.
        h       = f['Header'].attrs.get('HubbleParam')         # h = H0/(100km/s/Mpc)
        boxsize = f['Header'].attrs.get('BoxSize')             # L [cMph/h].
    return a,h,boxsize

def _count_part(part):
    return len(part["mass"])

def _mass_part(part):
    return np.sum(part["mass"])   
################################

def sersic_brightness(x,y,n=4,I=10,cntx=0,cnty=0,pa=0,q=1,Re=5.0):
    """
     x,y : N,M grid of coordinates
     n : Sersic index
     I : amplitude
     cntx,cnty : center coordinates
     pa: pointing angle (in angles)
     q : axis ratio
    """
    x    = to_dimless(x)
    cntx = to_dimless(cntx)
    y    = to_dimless(y)
    cnty = to_dimless(cnty)
    
    if q==1:
        # for some reason pa has effects even if q is 1 
        pa=0
    paRad = pa*np.pi/180
    # rotate the galaxy by the angle paRad
    X = np.cos(paRad)*(x-cntx)+np.sin(paRad)*(y-cnty)
    Y = -np.sin(paRad)*(x-cntx)+np.cos(paRad)*(y-cnty)
    # include elliptical isophotes
    #r = np.sqrt(x**2+y**2) #-> if q==1
    xt2difq2 = Y/(q*q)
    r = np.sqrt(X**2+Y*xt2difq2)
    # brightness at distance r
    bn = 1.992*n - 0.3271
    brightness = I*np.exp(-bn*((r/Re)**(1.0/n)-1.0))
    return brightness
    
class Sersic():
    # useful to lens an image:
    def __init__(self,I=10,cntx=0,cnty=0,pa=0,q=1,n=4):
        raise DeprecationWarning("Something is off w. the centering when considering ellipticity. \
        Use lenstronomy 'lenstronomy.LightModel.Profiles.sersic import SersicElliptic' for example")
        self.cntx = cntx
        self.cnty = cnty
        self.pa   = pa
        self.q    = q
        self.n    = n
        self.I    = I
    def image(self,x,y):
        return sersic_brightness(x,y,**self.__dict__)
