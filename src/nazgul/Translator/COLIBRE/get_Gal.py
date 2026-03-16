# extract from a given simulation suite a set of plausible lenses
# and select randomly one of these
import os
import numpy as np
from pathlib import Path

import unyt as u  # package used by swiftsimio to provide physical units
from swiftgalaxy import SWIFTGalaxy, SOAP
from swiftsimio import SWIFTDataset, cosmo_quantity

from nazgul.Translator import min_z,max_z,min_mass
from nazgul.Translator.COLIBRE.pathfinder import simsuite_name

# the following is, so far, not a variable
colibre_base_path = Path("/cosma8/data/dp004/colibre/Runs/")

# later on consider if this is the one we want:
std_sim = Path("L0025N0752")
std_subsim = Path("THERMAL_AGN_m5")
# note: the following is for now used both as the path for the output list in the colibre dataset, as well as in the Ringbearer data structure.
def find_simulation_dir(sim=std_sim,
                        subsim=std_subsim):
    # allows for more flexibility
    simulation_dir =Path(str(sim)+"/"+str(subsim))
    return simulation_dir
#std_simulation_dir = std_sim/std_subsim
#print("Todo: check if this is the correct simulation suite:",std_simulation_dir)

##### Snap/z map ####
def get_snap_z_map(sim=std_sim,
                    subsim=std_subsim):
    simulation_dir = find_simulation_dir(sim=sim,subsim=subsim)
    snap_z_map=colibre_base_path/simulation_dir/"output_list.txt"
    return snap_z_map
    
def Read_ZSnap(file_name):
    z = []
    snap_snip = []
    with open(file_name, 'r') as data:
        for line in data.readlines():
            if line[0]== "#":
                continue
            p = line.split(", ")
            z.append(float(p[0]))
            snap_snip.append(str(p[1]).replace("\n",""))
    return np.array(z),snap_snip

def get_kw_snap_z(sim=std_sim,
                  subsim=std_subsim):
    snap_z_map = get_snap_z_map(sim=sim,subsim=subsim)
    z_all,snap_snip = Read_ZSnap(snap_z_map)
    # we want to ignore snipshots (don't remember why, but most likely there are no soap gals)
    z,snap= [],[]
    kw_snap_z = {}
    for i in range(len(z_all)):
        if snap_snip[i]=="Snapshot":
            z.append(z_all[i])
            snap.append(f'{i:04d}')
            kw_snap_z[snap[-1]] = z[-1]
        elif snap_snip[i]=="Snipshot":
            _ = "Ignore Snipshots"
            #print("Ignore Snipshots")
    return kw_snap_z

def get_kw_z_snap(sim=std_sim,
                    subsim=std_subsim):
    kw_snap_z = get_kw_snap_z(sim=sim,subsim=subsim)
    #inverted kw
    kw_z_snap = {}
    for k in kw_snap_z:
        kw_z_snap[kw_snap_z[k]] = k
    return kw_z_snap
    
def get_z(snap,sim=std_sim,subsim=std_subsim):
    kw_snap_z = get_kw_snap_z(sim=sim,subsim=subsim)
    # format snap to integer
    int_snap = int(str(snap).lstrip("0"))
    # format it back to str w. leading 0 to match kw_snap_z
    snap = f"{int_snap:04d}"
    return kw_snap_z[snap]
    
def get_snap(z,sim=std_sim,subsim=std_subsim):
    kw_z_snap = get_kw_z_snap(sim=sim,subsim=subsim)
    # consider a continous z instead of the discreet version
    # works for discreet z as well
    key_z = min(kw_z_snap.keys(),key=lambda k:np.abs(k-float(z)))
    snap  = f'{kw_z_snap[key_z]}'
    return snap

def get_z_snap(z=None,snap=None,sim=std_sim,subsim=std_subsim):
    if z is None and snap is None:
        raise UserWarning("Give either z or snap")
    if z is None:
        z = get_z(snap,sim=sim,subsim=subsim)
    else:
        snap = get_snap(z,sim=sim,subsim=subsim)
    return z,snap
#####################       

def get_soap_cat(snap_str,
                 sim=std_sim,
                 subsim=std_subsim,
                 colibre_base_path=colibre_base_path):
    simulation_dir = find_simulation_dir(sim=sim,subsim=subsim)
    soap_catalogue_file = os.path.join(
        colibre_base_path,
        simulation_dir,
        "SOAP-HBT/halo_properties_"+snap_str+".hdf5",
    )
    #print("soap_catalogue_file",soap_catalogue_file)
    return soap_catalogue_file

def get_virtual_snapfile(snap_str,
                         sim=std_sim,
                         subsim=std_subsim,
                         colibre_base_path=colibre_base_path,
                         **kwargs):
    virtual_snapshot_file = os.path.join(
        str(colibre_base_path),
        str(sim),
        str(subsim),
        "SOAP-HBT/colibre_with_SOAP_membership_"+snap_str+".hdf5")
    #print("virtual_snapshot_file",virtual_snapshot_file)
    return virtual_snapshot_file


def get_rnd_snap(sim=std_sim,
                 subsim=std_subsim,
                 max_z=max_z,
                 min_z=min_z):
    """By sampling uniformily z"""
    max_z = float(max_z)
    min_z = float(min_z)
    min_mass = float(min_mass)
    z_rnd = np.random.uniform(min_z,max_z)
    snap  = get_snap(z_rnd,sim=sim,subsim=subsim)
    #z     = get_z(snap)
    return snap

def get_rnd_kw_swiftgal(sim=std_sim,
                        subsim=std_subsim,
                        max_z=max_z,
                        min_z=min_z,
                        min_mass=min_mass,
                        colibre_base_path=colibre_base_path,
                        soap_catalogue_file=None,
                        verbose=True):
    """ Return inputs for the SimPartGal of COLIBRE for a random
    galaxy within a range of redshift and a min mass
    """
    min_z = float(min_z)
    max_z = float(max_z)
    min_mass = float(min_mass)
    if verbose:
        print("We randomly select a redshift/snap:")
    snap_str = get_rnd_snap(sim=sim,
                            subsim=subsim,
                            max_z=max_z,
                            min_z=min_z)
    if verbose:
        print("Snap selected:",snap_str)
    z = get_z(snap_str,sim=sim,subsim=subsim)
    if verbose:
        print("z=",z)

    if soap_catalogue_file is None:
        soap_catalogue_file = get_soap_cat(snap_str=snap_str,
                                           sim=sim,
                                           subsim=subsim,
                                           colibre_base_path=colibre_base_path)
    sd = SWIFTDataset(soap_catalogue_file)
    # We have to define based on what we select the mass - or if we want other criteria
    selection_criteria = sd.spherical_overdensity_200_crit
    if verbose:
        print("As selection criteria we are taking the ",selection_criteria.group_name,", ie ",selection_criteria.group)
    m200c = selection_criteria.total_mass
    if verbose:
        print("And considering its total_mass")
        print(m200c)
    min_mass = cosmo_quantity(min_mass,u.Msun, comoving=True, scale_factor=sd.metadata.a, scale_exponent=0)
    if verbose:
        print("and selecting only galaxies w. mass less than ",min_mass.to("Msun").to_string())
        print(min_mass)
    candidates = np.argwhere(m200c> min_mass).squeeze()

    chosen_halo_index = np.random.choice(candidates)

    # for reproducibility
    inputs     = {"sim":sim,
                  "subsim":subsim,
                  "max_z":max_z,
                  "min_z":min_z,
                  "min_mass":min_mass,
                  "colibre_base_path":colibre_base_path,
                  "soap_catalogue_file":soap_catalogue_file}
    
    kw_rnd_gal = {"soap_catalogue_file":soap_catalogue_file,
                  "chosen_halo_index":chosen_halo_index,
                  #"candidates":candidates,
                  "snap_str":snap_str,
                  "inputs":inputs}
    return kw_rnd_gal
    
def _get_swiftgal(chosen_halo_index, 
                 virtual_snapshot_file,
                 soap_catalogue_file,
                 **kwargs # to ignore
                ):
    swift_gal = SWIFTGalaxy(
        virtual_snapshot_file,
        SOAP(
            soap_catalogue_file,
            soap_index=chosen_halo_index,
        ),
    )
    return swift_gal

def get_swiftgal(soap_index,
                 sim=std_sim,subsim=std_subsim,
                z=None,snap=None,
                colibre_base_path=colibre_base_path
                ):
    z,snap_str = get_z_snap(z=z,snap=snap,sim=sim,subsim=subsim)
    
    soap_catalogue_file = get_soap_cat(snap_str=snap_str,
                                           sim=sim,
                                           subsim=subsim,
                                           colibre_base_path=colibre_base_path)
    virtual_snapshot_file = get_virtual_snapfile(snap_str,
                         sim=sim,
                         subsim=subsim,
                         colibre_base_path=colibre_base_path)

    swift_gal = SWIFTGalaxy(
        virtual_snapshot_file,
        SOAP(
            soap_catalogue_file,
            soap_index=soap_index,
        ),
    )

    return swift_gal



def get_rnd_kw_gal(max_z=max_z,
                   min_z=min_z,
                   min_mass=min_mass,
                   sim=std_sim,
                   subsim=std_subsim,
                   colibre_base_path=colibre_base_path):
    min_z = float(min_z)
    max_z = float(max_z)
    min_mass = float(min_mass)
    kw_rnd_gal = get_rnd_kw_swiftgal(max_z=max_z,
                                     min_z=min_z,
                                     min_mass=min_mass,
                                     sim=sim,
                                     subsim=subsim,
                                     colibre_base_path=colibre_base_path)
    
    virtual_snapshot_file = get_virtual_snapfile(**kw_rnd_gal)
    kw_rnd_gal["virtual_snapshot_file"] = virtual_snapshot_file
    return kw_rnd_gal

"""
from nazgul.pathfinder import get_catdir

def _from_simdir_to_sim_subsim(simulation_dir=std_simulation_dir):
    simulation_dir= Path(simulation_dir)
    sim = simulation_dir.parent
    subsim = simulation_dir.name
    return sim,subsim
    
def get_rnd_kw_gal(simulation_dir=simulation_dir,
                          max_z=max_z,
                          min_z=min_z,
                          min_mass=min_mass,
                          colibre_base_path=colibre_base_path,
                         soap_catalogue_file=None,
                  check_prev=True,save_pkl=True):
    sim,subsim = _from_simdir_to_sim_subsim(simulation_dir)
    catdir = get_catdir(sim=sim,subsim=subsim,simsuite=simsuite)
    catfile = catdir/"galcat.dll"
    if check_prev:
        try:
            with open(catfile,"rb") as f:
                dill.load(f)
        except FileNotFoundError:
            print("Previous catalogue not found,

# not really used:
def _get_random_swiftgal(max_z=max_z,
                        min_z=min_z,
                        min_mass=min_mass,
                        simulation_dir=simulation_dir,
                        colibre_base_path=colibre_base_path):
    
    kw_rnd_gal    =  get_rnd_kw_swiftgal(max_z=max_z,
                        min_z=min_z,
                        min_mass=min_mass,
                        simulation_dir=simulation_dir,
                        colibre_base_path=colibre_base_path)
    rnd_swift_gal = _get_swiftgal(**kw_rnd_gal)
    
    return rnd_swift_gal,kw_rnd_gal
"""                
