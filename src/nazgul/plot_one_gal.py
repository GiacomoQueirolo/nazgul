#WIP
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.tools import mkdir
from nazgul.pathfinder import tmp_dir
from nazgul.project_gal import project_kw_parts
from nazgul.AMR2D_PLL import plot_AMR_cells,AMR_density_PLL,get_MDfromAMRcells_PLL
from nazgul.Translator.particle_galaxy import clip_coord
from nazgul.Translator.translator import Gal2kwMXYZ_part

def plot_gal(gl, save_to_tmp: bool = True):
    gal_dir = gl.gal_dir 
    xyz_dm  = gl.dm["coords"].T
    xyz_str = gl.stars["coords"].T
    xyz_gas = gl.gas["coords"].T
    xyz_bh  = gl.bh["coords"].T
    
    x_dm  = xyz_dm[0]
    x_str = xyz_str[0]
    x_gas = xyz_gas[0]
    x_bh  = xyz_bh[0]
    
    
    m_str = gl.stars["mass"]
    m_dm  = gl.dm["mass"]
    m_gas = gl.gas["mass"]
    m_bh  = gl.bh["mass"]
    
    # bh can be ignored
    
    b = 40
    plt.style.use('classic')
    plt.hist(x_str,bins=b,weights=m_str, color="yellow",label="stars",alpha=.3)
    plt.hist(x_gas,bins=b,weights=m_gas, color="violet",label="gas",alpha=.3)
    plt.hist(x_dm,bins=b,weights=m_dm,color="grey",label="dm",alpha=.3)
    plt.yscale("log")
    plt.ylabel(r"M [M$_\odot$]")
    plt.xlabel("X coord [Mpc]")
    plt.legend()
    plt.title("Mass Histogram For Different Particles of 1 EAGLE Gal.")
    plt.tight_layout()
    if save_to_tmp:
        nm = tmp_dir/"mHistGal1.png"
        print(f"Saving {nm}")
        plt.savefig(nm)
    nm = gal_dir/"mHistGal1.png"
    print(f"Saving {nm}")
    plt.savefig(nm)
    plt.close()
    
    
    xy_str = xyz_str[:-1]
    xy_dm  = xyz_dm[:-1]
    xy_gas = xyz_gas[:-1]
    xy_bh  = xyz_bh[:-1]
    dxy = 3*np.max(np.std(xy_dm,axis=1))
    xm,ym=np.mean(xy_dm,axis=1)

    fg,axes = plt.subplots(2,2)
    nbins = 60
    ax = axes[0][0]
    ax.set_title("Stars particles")
    #im0 = ax.scatter(*xy_str,c=np.log(m_str),alpha=.2,cmap="coolwarm_r",marker=".")
    hist,edgex,edgey = np.histogram2d(*xy_str,bins=nbins,weights=np.log10(m_str))
    extent = [edgex[0],edgex[-1],edgey[0],edgey[-1]]
    im0 = ax.imshow(hist.T,extent=extent,origin="lower",cmap="coolwarm_r")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(Star Mass)")

    ax = axes[0][1]
    ax.set_title("DM particles")
    #im0 = ax.scatter(*xy_dm,c=np.log(m_dm),alpha=.2,cmap="winter",marker=".")
    hist,edgex,edgey = np.histogram2d(*xy_dm,bins=nbins,weights=np.log10(m_dm))
    extent = [edgex[0],edgex[-1],edgey[0],edgey[-1]]
    im0 = ax.imshow(hist.T,extent=extent,origin="lower",cmap="winter")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(DM Mass)")

    ax = axes[1][0]
    ax.set_title("Gas particles")
    #im0 = ax.scatter(*xy_gas,c=np.log(m_gas),alpha=.2,cmap="coolwarm_r",marker=".")
    hist,edgex,edgey = np.histogram2d(*xy_gas,bins=nbins,weights=np.log10(m_gas))
    extent = [edgex[0],edgex[-1],edgey[0],edgey[-1]]
    im0 = ax.imshow(hist.T,extent=extent,origin="lower",cmap="coolwarm")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(Gas Mass)")
    
    ax = axes[1][1]
    ax.set_title("Blackholes particles")
    #im0 = ax.scatter(*xy_bh,c=np.log(m_bh),alpha=.2,cmap="coolwarm_r",marker=".")
    hist,edgex,edgey = np.histogram2d(*xy_bh,bins=nbins,weights=np.log10(m_bh))
    extent = [edgex[0],edgex[-1],edgey[0],edgey[-1]]
    im0 = ax.imshow(hist.T,extent=extent,origin="lower",cmap="coolwarm")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(BH Mass)")
    
    for axi in axes:
        for axii in axi:    
            axii.set_xlim(xm-dxy,xm+dxy)
            axii.set_ylim(ym-dxy,ym+dxy)
            axii.set_xlabel("X [Mpc]")
            axii.set_ylabel("Y [Mpc]")
    plt.tight_layout()
    if save_to_tmp:
        nm = tmp_dir/"PartDistrGal.png"
        print(f"Saving {nm}")
        plt.savefig(nm)
    
    nm = gal_dir/"PartDistrGal.png"
    print(f"Saving {nm}")
    plt.savefig(nm)
    plt.close()
    
def plot_AMR_density(gl,
                  max_particles=100,
                  min_area=0.1*u.kpc*u.kpc,
                  dens_thresh = 0.*u.Msun/(u.kpc**2),
                  verbose=True):
    """ 
    Compute and plot density Adaptive Mesh Refinement map split into the various particles
    input  : gal 
    returns: kw_2Ddens["MD_value"][u.Msun/(u.kpc**2),1] 
             kw_2Ddens["MD_coord"][arcsec,2]
             kw_2Ddens["AMR_cells"][cells,N]
    """

    print("ignoring BH - too few to make a AMR")
    types = ["stars","dm","gas"] #,"BH"]
    
    savedir = f"tmp/AMR_{gl.name}/"
    mkdir(savedir)
    for tp in types:
        # for now only considering index 0
        proj_index    = 0
        kw_parts      = Gal2kwMXYZ_part(gal,part_type=tp)
        kw_parts_proj = project_kw_parts(kw_parts,proj_index)
        Ms = np.asarray(kw_parts_proj["Ms"].to("Msun"))*u.Msun
        Xs = np.asarray(kw_parts_proj["Xs"].to("kpc"))*u.kpc
        Ys = np.asarray(kw_parts_proj["Ys"].to("kpc"))*u.kpc

        # units are stripped by numba - have to "reattach" them "by hand"
        AMR_cells = AMR_density_PLL(Xs,Ys,Ms, max_particles=max_particles, 
                                    min_area=min_area,dens_thresh=dens_thresh)
        # use parallelised version - faster 
        MD_coords,MD_value = get_MDfromAMRcells_PLL(AMR_cells) 
        # Note: all inputs are still in kpc
        kw_2Ddens = {"MD_value":MD_value,"MD_coords":MD_coords,"AMR_cells":AMR_cells}
        
        fig,ax = plot_AMR_cells(kw_2Ddens)
        nm = f"{savedir}/AMR_{tp}_proj{proj_index}.png"
        fig.savefig(nm)
        print(f"Saved {nm}") 
        plt.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot 2D mass distribution of the galaxy")
    
    raise NotImplementedError