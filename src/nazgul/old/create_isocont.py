# copied from create_mass_image.py (now old)
# we want to create iso-mass (and maybe iso-phot) contours of the galaxy

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
#from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from fnct import gal_dir,std_sim
from ParticleGalaxy import get_rnd_PG

#from PG_proj_part_hist import z_source_max,pixel_num,prep_Gal_projpath
from Gen_PM_PLL_AMR import pixel_num
from project_gal_AMR import prep_Gal_projpath

verbose      = True

#from Gen_PM_PLL import     cutoff_radius 
cutoff_radius = 100*u.kpc

from ParticleGalaxy import Gal2kwMXYZ 
from project_gal  import get_densmap,xyminmax

from scipy.ndimage import gaussian_filter
from python_tools.tools import to_dimless


def _plot_dens_map_hist(kw_parts,proj_index,pixel_num,cutoff_radius,sigma_smooth,nlevels):
    density = get_densmap(kw_parts,proj_index,pixel_num*3,
                          cutoff_radius=cutoff_radius,
                          cutoff_radius_dens=100*u.kpc,verbose=True)

    smooth_density = gaussian_filter(density.value, sigma=sigma_smooth)
    vmin, vmax = np.nanmin(smooth_density), np.nanmax(smooth_density)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels)
    xmin,xmax,ymin,ymax = list(np.array(xyminmax(cutoff_radius)).ravel())
    extent              = [xmin,xmax,ymin,ymax] 
    return {"smooth_density":smooth_density,"levels":levels,"extent":extent}
    
def plot_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,cutoff_radius=cutoff_radius,nlevels =14,
                       verbose=verbose,namefig=None,title=None,sigma_smooth=4):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)
    if nx==0:
        nx,ny = int(pixel_num), int(pixel_num)
    # given a projection, plot the density map

    kw_parts = Gal2kwMXYZ(Gal)
    
    print("NOTE: taking a small cutoff radius: ",cutoff_radius)

    # cutoff_radius_dens is used to have a first (quite alright)
    # estimate of the densest point
    kw_densplot = _plot_dens_map_hist(kw_parts,proj_index,pixel_num,cutoff_radius,sigma_smooth,nlevels)
    plt.close()
    if title is None:
        title = "Mass Density of Gal: "+str(Gal.Name)
    plt.title(title)
    plt.contour(kw_densplot["smooth_density"],extent=kw_densplot["extent"],levels=kw_densplot["levels"], 
                cmap=plt.cm.inferno,norm=LogNorm(),origin="lower")
    plt.colorbar(label=r"Density [M$_o$/kpc$^2$]")
    #plt.contour(np.log10(density),extent=extent,norm="log",cmap=plt.cm.gist_earth_r)
    #plt.contour(np.log10(density),extent=extent,cmap=plt.cm.inferno)
    plt.xlim(kw_densplot["extent"][:2])
    plt.ylim(kw_densplot["extent"][2:])
    plt.xlabel("kpc")
    plt.ylabel("kpc")
    
    if namefig is None:
        namefig = f"./tmp/cmi_densmap_proj_{proj_index}.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)
    return 0
def plot_densWzoom(Gal,proj_index=0,pixel_num=pixel_num,cutoff_radius=cutoff_radius,cutoff_radius_zoom=2*cutoff_radius,
                    nlevels =14,title_zoom=None,
                       verbose=verbose,namefig=None,title=None,sigma_smooth=4):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)
    if nx==0:
        nx,ny = int(pixel_num), int(pixel_num)
    # given a projection, plot the density map

    kw_parts = Gal2kwMXYZ(Gal)
    print("NOTE: taking a small cutoff radius: ",cutoff_radius)
    kw_densplot = _plot_dens_map_hist(kw_parts,proj_index,pixel_num,cutoff_radius,sigma_smooth,nlevels)
    print("NOTE: taking a zoomed in cutoff radius: ",cutoff_radius_zoom)
    kw_densplot_zoom = _plot_dens_map_hist(kw_parts,proj_index,pixel_num,cutoff_radius_zoom,sigma_smooth,nlevels)
    for kw in kw_densplot,kw_densplot_zoom:
        kw["image"] = kw.pop("smooth_density")

    if namefig is None:
        namefig = f"./tmp/cmi_densmap_proj_{proj_index}_zoom.png"
    kw_densplot["title"] = title
    kw_densplot["cutoff_radius"] = cutoff_radius
    kw_densplot_zoom["title"] = title_zoom
    kw_densplot_zoom["cutoff_radius"] = cutoff_radius_zoom
    plot_zoomed(kw_main=kw_densplot,kw_zoom=kw_densplot_zoom,namefig=namefig)
    return 0
    
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _plot_dens_map_hist(kw_parts,proj_index,pixel_num,cutoff_radius,sigma_smooth,nlevels):
    density = get_densmap(kw_parts,proj_index,pixel_num*3,
                          cutoff_radius=cutoff_radius,cutoff_radius_dens=100*u.kpc,verbose=True)

    smooth_density = gaussian_filter(density.value, sigma=sigma_smooth)
    vmin, vmax = np.nanmin(smooth_density), np.nanmax(smooth_density)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), nlevels)
    xmin,xmax,ymin,ymax = list(np.array(xyminmax(cutoff_radius)).ravel())
    extent              = [xmin,xmax,ymin,ymax] 
    return {"smooth_density":smooth_density,"levels":levels,"extent":extent}

def plot_zoomed(kw_main,kw_zoom,namefig):
    main_map = kw_main["image"]
    title_main = kw_main["title"]
    cutoff_radius = kw_main["cutoff_radius"]
    
    zoom_map = kw_zoom["image"]
    title_zoom = kw_zoom["title"]
    cutoff_radius_zoom = kw_zoom["cutoff_radius"]
    
    for kw in kw_main,kw_zoom:
        k = "cutoff_radius"
        del kw[k]
            
    # compute x,y,w,h of zoom
    pixn = len(main_map)
    # pix_size_zoom = pix_size_main *  cutoff_radius_zoom/cutoff_radius
    pixn_zoom = pixn*to_dimless(cutoff_radius_zoom/cutoff_radius) #float
    xcnt,ycnt = pixn/2., pixn/2.   # float
    w, h   = int(pixn_zoom/2.) , int(pixn_zoom/2.)      # width & height (int)
    x0, y0 = int(xcnt-(pixn_zoom/2.)), int(ycnt-(pixn_zoom/2.))     # lower-left corner (int)

    # --- Create figure with two subplots ---
    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Draw zoom rectangle -> coords rescaled to match extent
    xmin,xmax,ymin,ymax = kw_main["extent"]
    Nx,Ny = kw_main["image"].shape
    x_plot = xmin + (x0 / Nx) * (xmax - xmin)
    y_plot = ymin + (y0 / Ny) * (ymax - ymin)

    rect = Rectangle((x_plot, y_plot), w, h, fill=False, edgecolor="black", linewidth=2)
    ax_main.add_patch(rect)
    kwargs_clb = {'format': '%.1f'}
    for ax,kw,ttl in zip([ax_main,ax_zoom],[kw_main,kw_zoom],[title_main,title_zoom]):
        im = kw["image"]
        ttl = kw["title"]
        for k in "image","title":
            del kw[k]
        im_i = ax.contour(im,**kw, cmap=plt.cm.inferno,norm=LogNorm(), origin="lower")
        ax.set_title(ttl)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        try:
            fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa$",**kwargs_clb)
        except ValueError:
            print("Colobar not possible due to " +str(np.max(np.abs(im))))
            
        ax.set_xlim(kw["extent"][:2])
        ax.set_ylim(kw["extent"][2:])
        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")


    """
    # =======================================================
    # (3) Connector lines
    # =======================================================

    # Rectangle corners (main map)
    corners_main = [
        (x0, y0),               # lower-left
        (x0 + w, y0 + h)        # upper-right
    ]

    # Approximate corresponding corners in the zoom plot
    # (we use axis coordinate system for simplicity: 0–1)
    corners_zoom = [
        ax_zoom.transAxes.transform((0, 0.05)),    # attach near bottom-left
        ax_zoom.transAxes.transform((1, 0.05)),    # attach near top-left
    ]

    # Convert main image coordinates to figure coordinates
    for (xm, ym), (xz_fig, yz_fig) in zip(corners_main, corners_zoom):
        
        # main map: data coords → display coords → figure coords
        xm_fig, ym_fig = ax_main.transData.transform((xm, ym))

        # zoom map: display coords already → figure coords
        # (we need to convert display coords to figure coords for plotting lines)
        xz, yz = xz_fig, yz_fig

        # Finally transform display coords to figure coords
        inv_fig = fig.transFigure.inverted()
        x0f, y0f = inv_fig.transform((xm_fig, ym_fig))
        x1f, y1f = inv_fig.transform((xz, yz))

        # Draw line in figure coordinates
        line = Line2D([x0f, x1f], [y0f, y1f], transform=fig.transFigure, 
                      linewidth=1, color="black")
        fig.add_artist(line)
    """

    plt.tight_layout()
    print("Saving "+namefig)
    plt.savefig(namefig)


if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default="proj_part_hist")
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
                        default=True,action="store_false",help="if True, rerun code")

    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = True#args.verbose
    
    Gal = get_rnd_PG()
    Gal = prep_Gal_projpath(Gal)
    plot_dens_map_hist(Gal=Gal,pixel_num=pixel_num,verbose=verbose,cutoff_radius=cutoff_radius)
    
    print("Success")