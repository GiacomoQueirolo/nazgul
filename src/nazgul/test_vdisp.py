import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.get_res import load_whatever

from nazgul.stat_lenses import get_all_gallens
from nazgul.Translator import std_sim,std_simsuite
from nazgul.Translator.translator import PartGal
from nazgul.Translator.EAGLE.particle_galaxy import *
from nazgul.mount_doom.generate_gal_lens import GalLens
from nazgul.Translator.EAGLE.sql_connect import exec_query
from nazgul.Translator.EAGLE.get_gal_indexes import get_catpath
from nazgul.Translator.EAGLE.fnct import get_snap

from nazgul.configurations import min_z,max_z,min_mass

def get_vdisp(snap,Gn,SGn,**kwargs_query):
    if getattr(kwargs_query,"simsuite",std_simsuite)!=std_simsuite:
        raise NotImplementedError()
    try:
        cat_path = get_catpath(**kwargs_query)
        query_out = load_whatever(cat_path)
        if not "SVD" in query_out.keys():
            if "Vdisp" in query_out.keys():
                query_out["SVD"] = query_out["Vdisp"]
            else:
                raise RuntimeError(f"Query results do not contain velocity dispersion. Keys:{query_out.keys()}")
    except Exception as e:
        print(f"Failed to recover previous cat due to Error: {e}")
        print("Rerunning query")
        """myquery = "SELECT \
            gal.StellarVelDisp as SVD \
        FROM \
            %s_Subhalo as gal \
        WHERE \
            gal.Snapnum = %s and \
            gal.Mass > 1e12 and \
            gal.GroupNumber = %s and \
            gal.SubGroupNumber = %s"%(sim,snap,Gn,SGn)
        query_out = exec_query(myquery)"""
        if "simsuite" in kwargs_query:
            del kwargs_query["simsuite"]
        myQuery = get_query(**kwargs_query)
        query_out = exec_query(myQuery)
        
    vdisp_stars_all = query_out["SVD"]*u.km/u.s
    list_Gn   = query_out["Gn"]
    list_SGn  = query_out["SGn"]
    list_z    = query_out["z"]
    list_snap = np.array([int(get_snap(zi)) for zi in list_z])
    vdisp_stars = vdisp_stars_all[(list_Gn==Gn) & (list_SGn==SGn) & (list_snap==int(snap))]
    assert len(vdisp_stars)==1
    return vdisp_stars[0]
    
def get_tEsis(lensgal,**kwargs_query):
    lensgal.unpack()
    cosmo = lensgal.cosmo
    Ds    = cosmo.angular_diameter_distance(lensgal.z_source)
    Dls   = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.z_source)

    # we do not pass it, but we verify it's the same
    assert kwargs_query["sim"] == lensgal.Gal.sim
    vdisp_stars = get_vdisp(snap=lensgal.Gal.snap,
                            Gn=lensgal.Gal.Gn,
                            SGn=lensgal.Gal.SGn,
                            **kwargs_query)
    
    theta_E_sis = get_tE_sis(vdisp_stars,Dls=Dls,Ds=Ds)
    return theta_E_sis.value

def get_tE_sis(v_disp,Dls,Ds):
    vdis_c_ratio = v_disp/const.c
    theta_E_sis_radns = 4*np.pi*(vdis_c_ratio.value)*Dls/Ds
    theta_E_sis = theta_E_sis_radns.value*u.radian.to("arcsec")*u.arcsec
    return theta_E_sis    

def get_kw_tE(out_dll="tmp/del_theta_sis.dll",
              kw_get_gallens={},
              kwargs_query={}):
    lenses  = get_all_gallens(**kw_get_gallens)
    theta_E_list = []
    theta_sis_list = []
    zs = []
    for lensgal in lenses:
        lensgal.unpack()
        theta_E_sis = get_tEsis(lensgal,**kwargs_query)
        theta_sis_list.append(theta_E_sis)
        theta_E_list.append(lensgal.thetaE.value)
        zs.append(lensgal.z_lens)
        print(lensgal.thetaE)
        print(theta_E_sis)
        print("Ratio of theta_E computed vs theta_E_sis from vel. disp:",lensgal.thetaE/theta_E_sis)
    theta_sis_list = np.array(theta_sis_list)
    theta_E_list = np.array(theta_E_list)
    kw_tE = {"theta_E_sis":theta_sis_list,"theta_E":theta_E_list,"z_lens":zs}
    with open(out_dll,"wb") as f:
        dill.dump(kw_tE,f)

    return kw_tE
    
def comp_tE_vs_tEsis(reload=True,
                     out_dll = "tmp/del_theta_sis.dll",
                     name_plot = "tmp/tE_sis_vs_comp.png",
                     title = r"Compare $\theta_E$",
                    kw_get_gallens={},
                    kwargs_query={}):
    
    if reload:
        try:
            kw_tE = load_whatever(out_dll)
            print(f"Loaded prev. res. {out_dll}")
        except:
            print(f"Failed to load {out_dll}")
            reload = False
    if reload == False:
        kw_tE = get_kw_tE(kw_get_gallens=kw_get_gallens,
                          kwargs_query=kwargs_query,
                         out_dll=out_dll)
    theta_sis_list = kw_tE["theta_E_sis"]
    theta_E_list   = kw_tE["theta_E"]
    zs             = kw_tE["z_lens"]
    

    fig,axis = plt.subplots(1,2, figsize=(8, 4))
    plt.suptitle(title)
    ax = axis[0]
    sis_range = theta_sis_list.min(),theta_sis_list.max()
    tE_range  = theta_E_list.min(),theta_E_list.max()
    sis_11 = np.linspace(*sis_range,10)
    tE_11 = np.linspace(*sis_range,10)
    ax.plot(sis_11,tE_11,ls="--",c="k",label="1:1")
    im0 = ax.scatter(theta_sis_list,theta_E_list,marker=".",c=zs,cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"z$_{lens}$")
    ax.axis('equal')
    ax.set_ylabel(r"$\theta_E$")
    ax.set_xlabel(r"$\theta_E(v_{disp})$")
    ax.legend()
    
    ax = axis[1]
    x = np.arange(len(theta_sis_list))
    im0 = ax.scatter(x,theta_E_list/theta_sis_list,marker="x",c=zs,cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"z$_{lens}$")
    ax.plot(x,np.ones_like(x),ls="--",c="k",label="1")
    
    ax.set_xlabel(r"Gal")
    ax.set_ylabel(r"$\theta_E$/$\theta_E(v_{disp})$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(name_plot)
    print(f"Saving {name_plot}")
    plt.close()

if __name__=="__main__":
    print("SETUP FOR SEAGLE_I")
    kw_get_gallens = {"snaps":[25,26,27],
                     "sim":"RefL0050N0752"}
    kwargs_query   = {"sim":"RefL0050N0752",
    "min_hmr":1,                                   
    "min_z":0.1,#0.49,
    "max_z":0.3,#0.51,
    "min_vel_disp":120,
    "min_mass_stars":1.76e10*.6777}
    reload = False
    name_plot = "tmp/tE_sis_vs_comp_SEAGLEI.png"
    title = r"Compare $\theta_E$ (SEAGLE I)"
    comp_tE_vs_tEsis(reload=reload,
                     title=title,
                kw_get_gallens=kw_get_gallens,
                    kwargs_query=kwargs_query,
                    name_plot=name_plot
                    )
