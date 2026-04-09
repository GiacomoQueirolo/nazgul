# Plot all lenses
import numpy as np
from stat_lenses import get_all_gallens
from nazgul.mount_doom.lens_system import LensSystem
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def auto_grid(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_grid(images,savename,extents=None,cmap="hot",titles=None,label=None,suptitle=None):
    n = len(images)  # or any list of things to plot
    
    nrows, ncols = auto_grid(n)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    
    axes = axes.ravel()  # flatten for easy iteration
    
    for i, img in enumerate(images):
        if extents is not None:
            extent = extents[i]
        im0 = axes[i].imshow(img,origin="lower",extent=extent,cmap=cmap)
        axes[i].set_xlabel("X [kpc]")
        axes[i].set_ylabel("Y [kpc]")
        if titles is not None:
            axes[i].set_title(titles[i])
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical',label=label)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savename)
    print(f"Saved {savename}")
    
if __name__ == "__main__":

    lenses  = get_all_gallens()
    kappas  = []
    images  = []
    extents_arc = []
    extents_kpc = []
    gal_names   = []
    for l in lenses:
        kappas.append(l.kappa_map)
        kw_extents = l.kw_extents
        extents_kpc.append(kw_extents["extent_kpc"])
        extents_arc.append(kw_extents["extent_arcsec"])
        lns = LensSystem.from_GalLens(l)
        lns.setup()
        print("Very arbitrary moving the source 'by hand'")
        sig_source = lns.gallens.thetaE.value/3
        rad_source     = np.random.uniform(0,sig_source)
        phi_source     = np.random.uniform(0,2*np.pi)
        ra_source      = rad_source*np.cos(phi_source) 
        dec_source     = rad_source*np.sin(phi_source) 
        lns.update_source_position(ra_source,dec_source)
        images.append(lns.get_lensed_image())
        gal_names.append(l.Gal.name+" "+str(l.proj_index) )
    log_images = np.log10(np.array(images))
    plot_grid(log_images,"tmp/grid_gal_images.png",extents=extents_kpc,cmap="winter",titles=gal_names,label=r"log$_{10}$ flux [arbitrary]",suptitle="Lensed Images")
    plot_grid(kappas,"tmp/grid_gal_kappas.png",extents=extents_kpc,titles=gal_names,label=r"$\kappa$",suptitle="Convergence Map")
    