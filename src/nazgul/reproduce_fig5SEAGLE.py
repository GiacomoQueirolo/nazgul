# copy from plot_one_gal
# the aim is to reproduce Fig.5 of SEAGLE I 
# the  Surface density profiles of DM, stars, gas and the total mass of a typical ETG from EAGLE.
import sys
import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.interpolate import interp1d

from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.tools import mkdir,ensure_unit
from nazgul.pathfinder import tmp_dir
from nazgul.project_gal import project_kw_parts,dens_map_AMR,cells2SigRad

from nazgul.Translator.translator import PartGal,Gal2kwMXYZ
from nazgul.AMR2D_PLL import plot_AMR_cells,AMR_density_PLL,get_MDfromAMRcells_PLL
from nazgul.Translator.particle_galaxy import clip_coord

def Gal2MXYZ_part(Gal,part_type="stars"): 
    """Given the galaxy, return Masses (in Msun) and
    XY coords. of particles in kpc  centered around center
    """
    part = getattr(Gal,part_type) 
    # Particle masses
    
    Ms = part["mass"]*u.Msun #Msun
    
    # Particle pos
    Xs,Ys,Zs =  np.transpose(part["coords"]) *u.Mpc.to("kpc")*u.kpc #kpc

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

# Plot AMR density for different particle species

def Gal2kwMXYZ_part(Gal,part_type): 
    Ms, Xs,Ys,Zs = Gal2MXYZ_part(Gal,part_type=part_type)
    return {"Ms":Ms,"Xs":Xs,"Ys":Ys,"Zs":Zs}

def plot_AMR_density(gal,
                     proj_index    = 0,
                     z_source      = 2, # for now fixed
                     max_particles = 100,
                     min_area      = 0.1*u.kpc*u.kpc,
                     dens_thresh   = 0.*u.Msun/(u.kpc**2),
                     verbose=True):
    """ 
    Compute and plot density Adaptive Mesh Refinement map split into the various particles
    input  : gal 
    returns: kw_2Ddens["MD_value"][u.Msun/(u.kpc**2),1] 
             kw_2Ddens["MD_coord"][arcsec,2]
             kw_2Ddens["AMR_cells"][cells,N]
    """
    savedir  = f"tmp/AMR_{gal.name}/"
    mkdir(savedir)

    print("ignoring BH - too few to make a AMR")
    types        = ["stars","dm","gas"] #,"BH"]
    types_str    = ["Stars","DM","Gas"] #,"BH"]
    col_type     = ['red','green','blue'] 

    z_lens   = gal.z 
    arcXkpc  = gal.cosmo.arcsec_per_kpc_proper(z_lens)
    Dd       = gal.cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds       = gal.cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds      = gal.cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 

    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    
    kw_parts_all = Gal2kwMXYZ(gal)
    kw_parts_all_proj = project_kw_parts(kw_parts_all,proj_index)
    kw_2Ddens_all     = dens_map_AMR(kw_parts_proj=kw_parts_all_proj,
                                     max_particles=max_particles,
                                     min_area=min_area,
                                     dens_thresh=dens_thresh)
    figall,axall = plot_AMR_cells(kw_2Ddens_all)
    nm = f"{savedir}/AMR_full_proj{proj_index}.png"
    figall.savefig(nm)
    print(f"Saved {nm}") 
    r_all,Sigma_encl_all   = cells2SigRad(kw_2Ddens_all)
    r_all = r_all.to("kpc")

    Sigma_encl_all = Sigma_encl_all.to("Msun/kpc^2")
    Sigma_crit     = ensure_unit(Sigma_crit,Sigma_encl_all.unit)

    RE = np.interp(Sigma_crit.value, Sigma_encl_all.value[::-1], r_all[::-1].value)*r_all.unit


    fig2,ax2 = plt.subplots(1)
    ax2.plot(r_all,Sigma_encl_all/1e9,color='cyan',label="Total")

    ax_tmp = ax2.twiny() 

    for i_tp,tp in enumerate(types):
        kw_parts      = Gal2kwMXYZ_part(gal,part_type=tp)
        kw_parts_proj = project_kw_parts(kw_parts,proj_index)
        
        kw_2Ddens     = dens_map_AMR(kw_parts_proj=kw_parts_proj,
                                     max_particles=max_particles,
                                     min_area=min_area,
                                     dens_thresh=dens_thresh)
        # plot 2d dens distr.
        fig1,ax1 = plot_AMR_cells(kw_2Ddens)
        nm = f"{savedir}/AMR_{tp}_proj{proj_index}.png"
        fig1.savefig(nm)
        print(f"Saved {nm}") 

        # plot 1D Sigma
        r,Sigma_encl   = cells2SigRad(kw_2Ddens)
        r = r.to("kpc")
        Sigma_encl = Sigma_encl.to("Msun/kpc^2")
        ax2.plot(r,Sigma_encl/1e9,color=col_type[i_tp],label=types_str[i_tp])
    tE = RE*arcXkpc
    ax_tmp.plot(r*arcXkpc, np.zeros_like(r),alpha=0.)
    ax_tmp.tick_params(axis='x')

    ax2.set_title("Surface density profile (cnf. Fig 5 SEAGLE I)")
    ax2.axvline(RE.value,ls="--",c="k",label=r"R$_{\rm{Ein}}$="+f'{np.round(RE.value,2)} {RE.unit}={np.round(tE.value,2)}"')
    ax2.set_xlabel(f"Radius (R) [{r.unit}]")
    ax2.set_ylabel(r"$\Sigma$(R) [10$^9$"+str(Sigma_encl.unit)+"]")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax_tmp.set_xscale('log')

    ax_tmp.set_xlabel(f"Radius (R) ['']")
    ax2.legend()
    nm = f"{savedir}/Sigma_decomposed_proj{proj_index}.png"
    fig2.tight_layout()
    fig2.savefig(nm)

    print("DEBUG - theta_E,",tE,"tE w inverted conversion fact",RE/arcXkpc)
    print(f"Saved {nm}") 
    
    
if __name__=="__main__": 
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot 2D mass distribution of the galaxy and Fig 5 SEAGLE_1")
    #sim,Gn,SGn,snap = "RefL0025N0752",13,0,"25"
    sim,Gn,SGn,snap = "RefL0050N0752",8,0,"27"
    print("Using simulation: "+sim)
    print("Snap: "+snap)
    print("Galaxy: Gn",Gn,"SGn",SGn)
    
    Gal    = PartGal({"Gn":Gn,"SGn":SGn},simsuite="EAGLE",
                 sim=sim,
                 z=None,snap=snap,    # redshift or snap
                 M=None,Centre=None,
                 reload=False)
    Gal.run()
    plot_AMR_density(Gal)
    