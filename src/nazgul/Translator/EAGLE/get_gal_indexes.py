# Execute the query and get catalogues of galaxies 
# output a table of indexes to locate them, with 
# - coordinates
# - mass
# - redshift
# - Group and Subgroup

import dill
import warnings
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

from nazgul.pathfinder import std_simsuite,std_sim,get_sim_dir,get_catdir

from python_tools.tools import short_SciNot
from python_tools.get_res import load_whatever

from nazgul.configurations import min_z,max_z,min_mass

def get_gals(sim=std_sim,simsuite=std_simsuite,
             min_mass=min_mass,
             min_z=min_z,
             max_z=max_z,
             save_pkl=True,plot=True,check_prev=True,verbose=True,
            **kwargs_query):

    min_mass = short_SciNot(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    
    cat_path = get_catpath(min_mass=min_mass,\
                           min_z=min_z,max_z=max_z,\
                           sim=sim,simsuite=simsuite,**kwargs_query)
    
    # select higher masses bc 1) lenses 2) else we have too many points
    myQuery = get_query(sim=sim,min_mass=min_mass,\
                        min_z=min_z,max_z=max_z,**kwargs_query)
    
    # NOTE: center of mass is in comoving coord.(cMpc)
    # Execute
    
    #check_prev = False
    #save_pkl  = False
    
    if check_prev:
        found_prev = False
        if cat_path.exists():
            try:
                myData = load_whatever(cat_path)
                #formatting might be slightly diff.
                if myData["query"].replace(" ","") == myQuery.replace(" ",""):
                    
                    found_prev =True
                    if verbose:
                        print(f"Found previous pickled catalogue:\n{cat_path}")
                else:
                    if verbose:
                        print(f"Not same query in prev. cat.:\n{cat_path}\nRerunning and overwriting.")
            except Exception as e:
                print(f"Tried and failed to load previous results :{cat_path}\nBecause of {e}\nRerunning SQL query.")
                check_prev = False    
                    
        if not found_prev:
            check_prev = False            
    if not check_prev:
        # loads only if needed (avoid issues for tutorial)
        # TODO: implement more secure way to deal with password handling
        from nazgul.Translator.EAGLE.sql_connect import exec_query
        myData = exec_query(myQuery)
        # Store it/update 
        with open(cat_path,"wb") as f:
            dill.dump(myData,f)
        if verbose:
            print(f"Saving {cat_path}")
        
    if plot:
        _plot(myData)
    return myData

def get_query(sim=std_sim,min_mass=min_mass,min_z=min_z,max_z=max_z,
             min_mass_stars=None,min_vel_disp=None,min_hmr=None,
             AP_size = 10
             ):
    min_mass = short_SciNot(min_mass)
    min_z    = str(min_z)
    max_z    = str(max_z)
    mass_constr = f"gal.Mass > {min_mass}"
    if min_mass_stars is not None:
        mass_constr = f"AP.Mass_Star > {min_mass_stars}"
    vel_disp_constr = ""
    if min_vel_disp is not None:
        vel_disp_constr = f"and gal.StellarVelDisp > {min_vel_disp}"
    hmr_constr=""
    if min_hmr is not None:
        hmr_constr = f"and gal.HalfMassProjRad_Star > {min_hmr}"
    if AP_size!=10:
        warnings.warn("Aperture size != 10 pkpc - this differs from SEAGLE params")
    myQuery = f"SELECT \
        gal.GroupNumber as Gn, \
        gal.SubGroupNumber as SGn, \
        gal.Redshift as z, \
        gal.Mass as M, \
        gal.StellarVelDisp as SVD, \
        gal.CentreOfMass_x as CMx, \
        gal.CentreOfMass_y as CMy, \
        gal.CentreOfMass_z as CMz  \
    FROM \
        {sim}_Subhalo as gal, \
        {sim}_Aperture as AP \
    WHERE \
        AP.GalaxyID = gal.GalaxyID and \
        AP.ApertureSize = {AP_size} and \
        (gal.Redshift between {min_z} and {max_z}) and \
        {mass_constr}\
        {vel_disp_constr}\
        {hmr_constr}\
    ORDER BY \
        gal.Redshift"
    return myQuery

def get_catpath(sim=std_sim,simsuite=std_simsuite,
                min_mass = min_mass,min_z=min_z,max_z=max_z,**kwargs_query):
    min_mass = short_SciNot(min_mass)
    min_z    = short_SciNot(str(min_z))
    max_z    = short_SciNot(str(max_z))
    
    
    cat_name = f"CatGal_minM{min_mass}_minZ{min_z}_maxZ{max_z}"
    for k in kwargs_query:
        cat_name+=f"{k}{short_SciNot(kwargs_query[k])}"      
    cat_name+=".pkl"

    sim_path = get_catdir(sim=sim,simsuite=std_simsuite)
    cat_path = sim_path/cat_name 
    return cat_path
    
def _plot(myData):
    """Plot of informative statistic data
    """
    logMass = np.log(myData["M"])
    str_logMass = r'log$_{10}$M${_*}$[M$_{\odot}$]'
    zGal   = myData["z"]
    plt.hist(logMass)
    plt.title("Mass of Galaxies selected")
    plt.xlabel(str_logMass)
    plt.savefig("hist_gal_mass.png")
    plt.close()
    plt.hist(zGal)
    plt.title("Redshift of Galaxies selected")
    plt.xlabel(r'z')
    plt.savefig("hist_gal_z.png")
    plt.close()
    
    
    plt.scatter(zGal,logMass,marker=".")
    plt.title("Mass at redshift")
    plt.xlabel(r'z')
    plt.ylabel(str_logMass)
    plt.savefig("gal_mvsz.png")