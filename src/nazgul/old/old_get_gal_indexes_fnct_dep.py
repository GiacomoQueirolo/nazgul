# copy from plot_GSMF
# adapted to plot N of galaxies at different redshifts
# then output a table of indexes to use to get them, with 
# - coordinates
# - mass
# - redshift
# - Group and Subgroup (not really useful but anyway)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os,copy

from python_tools.get_res import load_whatever
from sql_connect import exec_query
from fnct import Galaxy,gal_dir,get_z,std_sim,gal_dir

def get_gals(sim=std_sim,min_mass = "1e12",min_z="0",max_z="2",save_pkl=True,pkl_name="massive_gals.pkl",plot=True,check_prev=True):
    pkl_path = f"{gal_dir}/{pkl_name}" 
     # select higher masses bc 1) lenses 2) else we have too many points
    myQuery = "SELECT \
        gal.GroupNumber as Gn, \
        gal.SubGroupNumber as SGn, \
        gal.Redshift as z, \
        gal.Mass as M, \
        gal.CentreOfMass_x as CMx, \
        gal.CentreOfMass_y as CMy, \
        gal.CentreOfMass_z as CMz  \
    FROM \
        %s_Subhalo as gal \
    WHERE \
        (gal.Redshift between %s and %s) and \
        gal.Mass > %s \
    ORDER BY \
        gal.Redshift"%(sim,min_z,max_z,min_mass)
    
    # NOTE: center of mass is in comoving coord.(cMpc)
    # Execute
    print("DEBUG - check that the pickling doesn't mess w. the data")
    check_prev = False
    save_pkl = False
    if check_prev:
        try:
            myData = load_whatever(pkl_path)
            #formatting might be slightly diff.
            if myData["query"].replace(" ","") != myQuery.replace(" ",""):
                raise UserWarning("Loaded previous results doesn't have the same query - rerunning and overwriting")
        except:
            print("Tried and failed to load previous results :"+pkl_path+"\nRerunning SQL query")
            check_prev = False
    if not check_prev:
        myData = exec_query(myQuery)
    if plot:
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
    """
    if save_pkl and not check_prev:
        with open(pkl_path,"wb") as f:
            pickle.dump(myData,f)
        print("Saving "+pkl_path)
    """
    return myData


def get_rnd_gal(sim=std_sim,min_mass = "1e12",min_z="0",max_z="2",pkl_name="massive_gals.pkl",check_prev=True,save_pkl=True,reuse_previous=True):
    if reuse_previous:
        # note: this way we will always use the same
        # TODO: randomly select within the acceptable ones
        for snap_dir in os.listdir(gal_dir):
            if "snap_" in snap_dir:
                snap = snap_dir.split("_")[1].split("/")[0]
                z_gl = get_z(int(snap))
                if z_gl<int(max_z) and z_gl>int(min_z):
                    for gl in os.listdir(gal_dir+"/"+snap_dir):
                        print(gal_dir+"/"+snap_dir+"/"+gl)
                        Gal = load_whatever(gal_dir+"/"+snap_dir+"/"+gl)
                        if Gal.M_tot>float(min_mass):
                            print("Found previous Gal "+str(Gal))
                            return Gal
        print("Previous Gal not found")
        reuse_previous = False
    if not reuse_previous:
        data  = get_gals(sim=sim,min_mass=min_mass,max_z=max_z,min_z=min_z,pkl_name=pkl_name,check_prev=check_prev,plot=False,save_pkl=save_pkl)
        Gal = _get_rnd_gal(data)
    return Gal


def _get_rnd_gal(data):
    index = np.arange(len(data["z"]))
    rnd_i = np.random.choice(index)
    kw = {}
    for k in data.keys():
        if k=="query" or k=="sim":
            kw[k] = data[k]
        else:
            kw[k] = data[k][rnd_i]

    Gal = Galaxy(**kw)
    return Gal

