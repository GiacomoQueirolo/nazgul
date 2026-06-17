# copy from plot_one_gal
# the aim is to reproduce Fig.5 of SEAGLE I 
# the  Surface density profiles of DM, stars, gas and the total mass of a typical ETG from EAGLE.
import sys
import warnings
import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const

from python_tools.tools import mkdir,ensure_unit
from python_tools.get_res import load_whatever
from nazgul.project_gal import project_kw_parts,dens_map_AMR,cells2SigRad

from nazgul.AMR2D_PLL import plot_AMR_cells
from nazgul.Translator.translator import PartGal,Gal2kwMXYZ,Gal2kwMXYZ_part,get_sim_func
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.plot_one_gal import plot_gal

base_colors = ["red","green","blue","yellow","dark","magenta","cyan",
               "darkorange","darkviolet","lawngreen","violet"] 
# Plot AMR density for different particle species
def plot_AMR_densityXpart(Gal,
                     proj_index    = 0,
                     z_source      = 2, 
                     max_particles = 100,
                     min_area      = 0.1*u.kpc*u.kpc,
                     dens_thresh   = 0.*u.Msun/(u.kpc**2),
                     scale_tE_cutout = 10,
                     savedir       = None,
                     part_thresh   = 1e4, # min n* of particles to be plotted
                     verbose=True):
    """ 
    Compute and plot density Adaptive Mesh Refinement map split into the various particles
    """
    if not savedir:
        savedir  = f"{Gal.gal_dir}/Plots/"
    mkdir(savedir)

    z_lens   = Gal.z 
    arcXkpc  = Gal.cosmo.arcsec_per_kpc_proper(z_lens)
    Dd       = Gal.cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds       = Gal.cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds      = Gal.cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    

    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    
    kw_parts_all      = Gal2kwMXYZ(Gal)
    kw_parts_all_proj = project_kw_parts(kw_parts_all,proj_index)

    
    kw_2Ddens_all     = dens_map_AMR(kw_parts_proj=kw_parts_all_proj,
                                     max_particles=max_particles,
                                     min_area=min_area,
                                     dens_thresh=dens_thresh,clip=True)
    # free memory
    del kw_parts_all,kw_parts_all_proj
    
    r_all,Sigma_encl_all   = cells2SigRad(kw_2Ddens_all)
    r_all = r_all.to("kpc")

    Sigma_encl_all = Sigma_encl_all.to("Msun/kpc^2")
    Sigma_crit     = ensure_unit(Sigma_crit,Sigma_encl_all.unit)

    RE = np.interp(Sigma_crit.value, Sigma_encl_all.value[::-1], r_all[::-1].value)*r_all.unit
    tE = RE*arcXkpc

    
    cutout_arcs = tE*scale_tE_cutout
    cutout_kpc  = RE*scale_tE_cutout
    kw_extents = {"extent_arcsec":[-cutout_arcs.value,cutout_arcs.value,-cutout_arcs.value,cutout_arcs.value],
                  "extent_kpc":[-cutout_kpc.value,cutout_kpc.value,-cutout_kpc.value,cutout_kpc.value]}
    
    # Plotting all the density
    
    figall,axall = plot_AMR_cells(kw_2Ddens_all,kw_extents=kw_extents)
    figall.suptitle("AMR of total mass projection")
    nm = f"{savedir}/AMR_full_proj{proj_index}.png"
    figall.savefig(nm)
    print(f"Saved {nm}") 
    plt.close(figall)
    fig2,ax2 = plt.subplots(1)
    ax2.plot(r_all,Sigma_encl_all/1e9,color='cyan',label="Total")

    ax_tmp = ax2.twiny() 
    part_types = get_sim_func(Gal.simsuite,"part_type_list")
    i = 0
    for tp in part_types:
        if "bh" in tp.lower() or "hole" in tp.lower():
            warnings.warn("Ignoring BH particles- too few to make an AMR")
            continue
        kw_parts      = Gal2kwMXYZ_part(Gal,part_type=tp)
        # consider the case that there are too few particles and skip it
        if len(kw_parts["Xs"])<part_thresh:
            warnings.warn(f"Less than {part_thresh} {tp} particles - skipping")
            continue
        
        kw_parts_proj = project_kw_parts(kw_parts,proj_index)
        kw_2Ddens     = dens_map_AMR(kw_parts_proj=kw_parts_proj,
                                     max_particles=max_particles,
                                     min_area=min_area,
                                     dens_thresh=dens_thresh,
                                     clip=True)
        # free memory
        del kw_parts,kw_parts_proj
        del kw_2Ddens["MD_coords"]
        kw_2Ddens["MD_coords"] = kw_2Ddens_all["MD_coords"]
        # plot 2d dens distr.
        fig1,ax1 = plot_AMR_cells(kw_2Ddens,kw_extents=kw_extents)
        fig1.suptitle(f"AMR of mass projection of {tp} particles")
        nm = f"{savedir}/AMR_{tp}_proj{proj_index}.png"
        fig1.savefig(nm)
        plt.close(fig1)
        print(f"Saved {nm}") 

        # plot 1D Sigma
        r,Sigma_encl   = cells2SigRad(kw_2Ddens)
        r = r.to("kpc")
        Sigma_encl = Sigma_encl.to("Msun/kpc^2")
        ax2.plot(r,Sigma_encl/1e9,color=base_colors[i],label=tp)
        i+=1
    ax_tmp.plot(r_all.to("kpc")*arcXkpc, np.zeros_like(r_all),alpha=0.)
    ax_tmp.tick_params(axis='x')

    ax2.set_title("Surface density profile (cnf. Fig 5 SEAGLE I)")
    ax2.axvline(RE.value,ls="--",c="k",label=r"R$_{\rm{Ein}}$="+f'{np.round(RE.value,2)} {RE.unit}={np.round(tE.value,2)}"')
    ax2.set_xlabel(f"Radius (R) [{r.unit}]")
    ax2.set_ylabel(r"$\Sigma$(R) [10$^9$"+str(Sigma_encl.unit)+"]")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax_tmp.set_xscale('log')

    ax_tmp.set_xlabel(f"Radius (R) ['']")
    ax2.legend(loc="upper right")
    nm = f"{savedir}/Sigma_decomposed_proj{proj_index}.png"
    fig2.tight_layout()
    fig2.savefig(nm)
    plt.close(fig2)
    plt.close("all")
    print(f"Saved {nm}") 
    
    
if __name__=="__main__": 
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot 2D mass distribution of the galaxy and Fig 5 SEAGLE_1")
    parser.add_argument('-snap','--snap',type=int,dest="snap",default=27,help=f"Snap to consider")
    parser.add_argument('-Gn',type=int,dest="Gn",default=8, help=f"Galaxy Number (Gn) to consider")
    parser.add_argument('-SGn',type=int,dest="SGn",default=0,help=f"Sub-Galaxy Number (SGn) to consider")
    parser.add_argument('-prj','--proj_index',type=int,dest="proj_ind",default=0,help=f"Projection index")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")

    #sim,Gn,SGn,snap = "RefL0025N0752",13,0,"25"
    args      = parser.parse_args()
    snap      = str(args.snap)
    Gn        = args.Gn
    SGn       = args.SGn
    proj_ind  = args.proj_ind
    sim       = args.sim
    subsim    = args.subsim
    simsuite  = args.simsuite

    print("Using simulation: "+sim)
    print("Snap: "+snap)
    print("Galaxy: Gn",Gn,"SGn",SGn)
    
    Gal    = PartGal({"Gn":Gn,"SGn":SGn},
                     simsuite=simsuite,
                     sim=sim,
                     subsim=subsim,
                     z=None,snap=snap,    # redshift or snap
                     M=None,Centre=None,
                     reload=False)
    Gal.run()
    plot_AMR_densityXpart(Gal,proj_index=proj_ind)
    plot_gal(Gal)
    