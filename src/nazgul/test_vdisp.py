import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from mpl_toolkits.axes_grid1 import make_axes_locatable


from nazgul.stat_lenses import get_all_gallens
from nazgul.Translator.translator import PartGal
from nazgul.Translator.EAGLE.particle_galaxy import *
from nazgul.mount_doom.generate_gal_lens import GalLens
from nazgul.Translator.EAGLE.sql_connect import exec_query

def get_vdisp(sim,snap,Gn,SGn):
    myquery = "SELECT \
        gal.StellarVelDisp as Vdisp \
    FROM \
        %s_Subhalo as gal \
    WHERE \
        gal.Snapnum = %s and \
        gal.Mass > 1e12 and \
        gal.GroupNumber = %s and \
        gal.SubGroupNumber = %s"%(sim,snap,Gn,SGn)
    query_out = exec_query(myquery)
    vdisp_stars = query_out["Vdisp"]*u.km/u.s
    return vdisp_stars
    
def get_tEsis(lensgal):
    lensgal.unpack()
    cosmo = lensgal.cosmo
    Ds    = cosmo.angular_diameter_distance(lensgal.z_source)
    Dls   = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.z_source)

    vdisp_stars = get_vdisp(lensgal.Gal.sim,lensgal.Gal.snap,lensgal.Gal.Gn,lensgal.Gal.SGn)
    
    vdis_c_ratio = vdisp_stars/const.c
    theta_E_sis_radns = 4*np.pi*(vdis_c_ratio.value)*Dls/Ds
    theta_E_sis = theta_E_sis_radns.value*u.radian.to("arcsec")*u.arcsec
    return theta_E_sis.value

def get_kw_tE(out_dll="tmp/del_theta_sis.dll"):
    lenses  = get_all_gallens()
    theta_E_list = []
    theta_sis_list = []
    zs = []
    for lensgal in lenses:
        lensgal.unpack()
        theta_E_sis = get_tEsis(lensgal)
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
    
def comp_tE_vs_tEsis(reload=True):
    out_dll = "tmp/del_theta_sis.dll"
    if reload:
        try:
            kw_tE = load_whatever(out_dll)
        except:
            print(f"Failed to load {out_dll}")
            reload = False
    if reload == False:
        kw_tE = get_kw_tE()
    theta_sis_list = kw_tE["theta_E_sis"]
    theta_E_list   = kw_tE["theta_E"]
    zs             = kw_tE["z_lens"]
    

    fig,axis = plt.subplots(1,2, figsize=(8, 4))
    plt.suptitle(r"Compare $\theta_E$")
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
    nm = "tmp/tE_sis_vs_comp.png"
    plt.tight_layout()
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close()

if __name__=="__main__":
    comp_tE_vs_tEsis()