from pathlib import Path
from python_tools.tools import mkdir
"""
New data structure:
RingBearer
    |_ Sims Suite  (e.g EAGLE)
        |_ Specific Res. Sim  (e.g RefL0025N0752)
            |_Snapshots   (e.g snap1)                                                  |_ CatGal
                |_GnSgn                                    |_ParticleData                   |_pkl
                     |_ Gal |_Proj |_Lens                         |_hdf5
                         |     |       |_ Algos
                         |     |            |_ Sub |_Dom
                         |     |_pkl            |     |_pkl
                         |_pkl                  |_pkl
"""

std_data_dir = Path("./RingBearer")
std_simsuite = "EAGLE"
std_sim      = "RefL0025N0752"
std_simsuite_dir = std_data_dir/std_simsuite # which simulation suite
std_sim_dir      = std_simsuite_dir/std_sim # which simulation

# use the following simulation only as test case
test_sim = "RefL0012N0188"
# used for tutorial -linked ONLY snap 20 of test_sim
tutorial_sim = "RefTuto"

# path to LensPop directory
LensPop_dir = Path("./LensPop/LensPop")

def get_simsuite_dir(simsuite=std_simsuite,data_dir=std_data_dir):
    data_dir     = Path(data_dir)
    simsuite_dir = data_dir/simsuite # which simulation suite
    return simsuite_dir

def get_sim_dir(sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    simsuite_dir = get_simsuite_dir(simsuite=simsuite,
                                    data_dir=data_dir)
    sim_dir      = simsuite_dir/sim # which simulation
    return sim_dir

def get_catdir(sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    sim_path = get_sim_dir(sim=sim,simsuite=std_simsuite,data_dir=data_dir)
    catdir = sim_path/"CatGal"
    mkdir(catdir)
    return catdir

def get_snap_dir(snap,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where the simulation particle data is stored
    """
    sim_dir = get_sim_dir(sim=sim,
                          simsuite=simsuite,data_dir=data_dir)
    snap     = str(snap).zfill(3)
    snap_dir = sim_dir/f"snap_{snap}"
    return snap_dir

def get_part_dir(snap=None,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where the simulation particle data is stored
    """
    part_dir_name = "ParticleData"
    if snap is not None:
        snap_dir = get_snap_dir(snap=snap,
                                sim=sim,simsuite=simsuite,
                                data_dir=data_dir)
        part_dir = snap_dir/part_dir_name
    else:
        sim_dir  = get_sim_dir(sim=sim,
                               simsuite=simsuite,data_dir=data_dir)
        
        part_dir = sim_dir/f"snap*/{part_dir_name}"
    return part_dir



def get_gal_maindir(Gn,SGn,snap,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where all galaxy results are stored
    """
    snap_dir = get_snap_dir(snap=snap,
                            sim=sim,simsuite=simsuite,
                            data_dir=data_dir)
    galname  = f"Gn{int(Gn)}SGn{int(SGn)}"
    gal_dir  = snap_dir/galname
    return gal_dir

def get_gal_dir(Gn,SGn,snap,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where particle galaxy is stored
    """
    gal_maindir = get_gal_maindir(Gn=Gn,SGn=SGn,snap=snap,
                                  sim=sim,simsuite=simsuite,data_dir=data_dir)
    gal_dir     = gal_maindir/"Gal"
    return gal_dir


nm_proj_dir = "Projection"
def get_proj_dir(Gn,SGn,snap,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where projection of the galaxy is stored
    """
    gal_maindir = get_gal_maindir(Gn=Gn,SGn=SGn,snap=snap,
                                  sim=sim,simsuite=simsuite,data_dir=data_dir)
    proj_dir     = gal_maindir/nm_proj_dir
    return proj_dir

def get_proj_dir_from_galdir(galdir):
    galdir      = Path(galdir)
    gal_maindir = galdir.parent
    proj_dir    = gal_maindir/nm_proj_dir
    return proj_dir   

def get_lens_maindir(Gn,SGn,snap,sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where lens computations of the galaxy are stored (main dir)
    """
    gal_maindir = get_gal_maindir(Gn=Gn,SGn=SGn,snap=snap,
                                  sim=sim,simsuite=simsuite,data_dir=data_dir)
    lens_dir     = gal_maindir/"Lens"
    return lens_dir

def get_lens_subdir(Gn,SGn,snap,
                    subdir="./",
                    sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    Where lens computations of the galaxy are stored 
    (sub dir - dep. on algorithm used, by default == main lens dir)
    """
    lens_maindir = get_lens_maindir(Gn=Gn,SGn=SGn,snap=snap,
                                  sim=sim,simsuite=simsuite,data_dir=data_dir)
    lens_subdir  = lens_maindir/subdir 
    return lens_subdir

nm_lowdir = "Sub"
def get_lens_lowdir(Gn,SGn,snap,subdir="./",
                    sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    lens computation dir (low-level, only particles)
    """
    lens_subdir = get_lens_subdir(Gn,SGn,snap,
                    subdir=subdir,
                    sim=sim,simsuite=simsuite,data_dir=data_dir)
    return  lens_subdir/nm_lowdir

def get_lens_lowdir_from_galdir(galdir):
    galdir      = Path(galdir)
    gal_maindir = galdir.parent
    low_dir     = gal_maindir/nm_lowdir
    return low_dir     

nm_highdir = "Dom"
def get_lens_highdir(Gn,SGn,snap,subdir="./",
                    sim=std_sim,simsuite=std_simsuite,data_dir=std_data_dir):
    """
    lens computation dir (high-level, lens model) 
    """
    lens_subdir = get_lens_subdir(Gn,SGn,snap,
                    subdir=subdir,
                    sim=sim,simsuite=simsuite,data_dir=data_dir)
    return  lens_subdir/nm_highdir
    
def get_lens_highdir_from_galdir(galdir):
    galdir      = Path(galdir)
    gal_maindir = galdir.parent
    high_dir    = gal_maindir/nm_highdir
    return high_dir   
    